# src/openCHA/tasks/ambiguity/diagnostic_ambiguity_task.py

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from pydantic import PrivateAttr

from ..task import BaseTask


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

TOPK_DIAG = 8
MAX_QUESTIONS = 3
MIN_MATCH_CAP = 2
HARD_FILTER_ABSENT = True
QUERY_TIMEOUT = 5.0
FT_INDEX = "term_name_ft"

# POMDP stopping threshold
# δ = 0.05 means agent answers when P(top hypothesis) > 0.95
# Grounded in Howard (1966) Value of Information:
# under zero elicitation cost, act when posterior risk < δ
DELTA = 0.05

# Legacy parameters kept only for _execute() backward compatibility
DEFAULT_EPSILON = 0.01
DEFAULT_T_MAX = 5


# ─────────────────────────────────────────────────────────
# Symptom normalization
# ─────────────────────────────────────────────────────────

NORMALIZE: Dict[str, str] = {
    "stuffy nose": "nasal congestion",
    "runny nose": "nasal congestion",
    "runny/stuffy nose": "nasal congestion",
    "short of breath": "shortness of breath",
    "shortness of breath": "shortness of breath",
    "shortness constipation breath": "shortness of breath",
    "trouble breathing": "dyspnea",
    "yellow mucus": "sputum production",
    "colored sputum": "sputum production",
    "body aches": "body ache",
    "sore throat": "pain in throat",
    "swollen lymph nodes": "enlarged lymph nodes",
    "swollen tonsils": "enlarged tonsil",
    "decreased appetite": "decrease in appetite",
    "joint pain": "pain of joint",
    "chest pressure": "pressure in chest",
    "sleep disturbances": "disturbance in sleep behavior",
    "cold intolerance": "intolerant of cold",
    "mood swing": "mood swings",
    "ear pressure": "aural pressure",
    "pain with bright lights": "eyes sensitive to light",
    "sneezing fits": "sneezing",
    "mucus secretion": "abnormal sputum",
    "mucus": "abnormal sputum",
    "reduced sex drive": "reduced libido",
    "inability to focus/concentrate": "poor concentration",
    "burning pee": "dysuria",
    "pain when peeing": "dysuria",
}


def _normalize(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    if t.endswith(")") and "(" in t:
        base = t[:t.rfind("(")].strip()
        if base:
            t = base
    return NORMALIZE.get(t, t)


def _uniform_entropy(n: int) -> float:
    return math.log2(n) if n > 1 else 0.0


def _parse_llm_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _ig(candidates: List[Dict[str, Any]], sx_cui: str) -> float:
    n = len(candidates)
    if n <= 1:
        return 0.0

    h_yes = [c for c in candidates if sx_cui in c["all_sym"]]
    h_no  = [c for c in candidates if sx_cui not in c["all_sym"]]

    p_yes = len(h_yes) / n
    p_no  = len(h_no)  / n
    h_after = (p_yes * _uniform_entropy(len(h_yes)) +
               p_no  * _uniform_entropy(len(h_no)))
    return _uniform_entropy(n) - h_after


def _is_bad_diagnosis_name(name: str) -> bool:
    n = (name or "").strip().lower()
    bad_substrings = [
        "immunodeficiency", "history of", "h/o", "defect",
        "syndrome", "classified elsewhere", "unspecified", "nos",
        "covid19-related", "ciliary dyskinesia",
        "mucopolysaccharidosis",
        "anti-glomerular basement membrane",
        "interstitial lung disease",
        "nodular lymphoid hyperplasia",
    ]
    if any(x in n for x in bad_substrings):
        return True
    if name.isupper() and len(name) > 10:
        return True
    return False


def _is_bad_question_symptom(name: str) -> bool:
    n = (name or "").strip().lower()
    bad_substrings = [
        "h/o", "history of", "defect", "ground glass",
        "opacity", "chronic", "impaired induction",
        "situs inversus", "lacrimal duct",
        "upper airway inflammation",
    ]
    return any(x in n for x in bad_substrings)


# ─────────────────────────────────────────────────────────
# Cypher queries
# ─────────────────────────────────────────────────────────

CY_CANDIDATES = """
UNWIND $symptom_cuis AS scui
MATCH (d:Diagnosis)-[:HAS_SYMPTOM]->(s {cui: scui})
WHERE NOT d.name STARTS WITH '['
  AND NOT toLower(d.name) CONTAINS 'unspecified'
  AND NOT toLower(d.name) CONTAINS 'nos'
WITH d, collect(DISTINCT s.cui) AS matched, size($symptom_cuis) AS reported
WITH d, matched, reported, size(matched) AS k
WHERE k >= $min_match
MATCH (d)-[:HAS_SYMPTOM]->(allS)
WITH d, matched, reported, k, collect(DISTINCT allS.cui) AS all_sym
WHERE size(all_sym) >= 1
RETURN d.cui AS cui,
       d.name AS name,
       k,
       size(all_sym) AS total,
       all_sym,
       (1.0 * k / reported) AS match_ratio
ORDER BY k DESC, match_ratio DESC, total ASC
LIMIT $topk
"""

CY_DISTINGUISHING = """
UNWIND $dx_cuis AS dcui
MATCH (d:Diagnosis {cui: dcui})-[:HAS_SYMPTOM]->(s)
WHERE NOT s.cui IN $known
RETURN DISTINCT s.cui AS cui, coalesce(s.name, s.cui) AS name
LIMIT $limit
"""

CY_NAME = """
MATCH (c {cui: $cui})
RETURN coalesce(c.name, $cui) AS name
LIMIT 1
"""

# IS_A family query — returns immediate clinical family for a diagnosis
CY_FAMILY = """
MATCH (d:Diagnosis {cui: $cui})-[:IS_A]->(p:Diagnosis)
WHERE NOT p.hierarchy_only IS NULL OR p.hierarchy_only = true
RETURN p.cui AS cui, p.name AS name
LIMIT 5
"""

# IS_A family for a set of diagnoses — finds family with majority coverage.
# Requires at least ceil(n/2) candidates to share the family.
# Orders by coverage (most members first), then name length (more specific first).
CY_SHARED_FAMILY = """
UNWIND $dx_cuis AS dcui
MATCH (d:Diagnosis {cui: dcui})-[:IS_A]->(p:Diagnosis)
WITH p.cui AS family_cui, p.name AS family_name,
     collect(dcui) AS members
WITH family_cui, family_name, members,
     size(members) AS coverage
WHERE coverage >= $min_coverage
  AND NOT toLower(family_name) CONTAINS 'face'
  AND NOT toLower(family_name) CONTAINS 'head'
  AND NOT toLower(family_name) CONTAINS 'disorder of'
  AND NOT toLower(family_name) CONTAINS 'finding'
RETURN family_cui, family_name, coverage
ORDER BY coverage DESC, size(family_name) DESC
LIMIT 1
"""


# ─────────────────────────────────────────────────────────
# Belief state helpers
# ─────────────────────────────────────────────────────────

def _belief_init(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Initialize uniform belief distribution over candidates.
    b(h) = 1/|H| for all h in H.
    """
    n = len(candidates)
    if n == 0:
        return {}
    p = 1.0 / n
    return {c["cui"]: p for c in candidates}


def _belief_update(
    belief: Dict[str, float],
    candidates: List[Dict[str, Any]],
    sx_cui: str,
    answer: bool,
) -> Dict[str, float]:
    """
    Bayes rule update after observing symptom answer.

    If answer=True:  keep only candidates that HAVE sx_cui
    If answer=False: keep only candidates that LACK sx_cui

    Renormalize after update.
    Grounded in standard Bayesian inference for binary observations.
    """
    new_belief = {}
    for c in candidates:
        cui = c["cui"]
        has_sx = sx_cui in c["all_sym"]
        if answer and has_sx:
            new_belief[cui] = belief.get(cui, 0.0)
        elif (not answer) and (not has_sx):
            new_belief[cui] = belief.get(cui, 0.0)

    total = sum(new_belief.values())
    if total > 0:
        return {cui: p / total for cui, p in new_belief.items()}
    return belief  # no update if degenerate


def _max_belief(belief: Dict[str, float]) -> float:
    """Return max(b) — the probability of the most likely hypothesis."""
    return max(belief.values()) if belief else 0.0


def _top_belief_cui(belief: Dict[str, float]) -> Optional[str]:
    """Return CUI of the most probable hypothesis."""
    if not belief:
        return None
    return max(belief, key=belief.__getitem__)


# ─────────────────────────────────────────────────────────
# Main task class
# ─────────────────────────────────────────────────────────

class DiagnosticAmbiguityTask(BaseTask):
    name:        ClassVar[str]       = "diagnostic_ambiguity"
    chat_name:   ClassVar[str]       = "DiagnosticAmbiguityAgent"
    description: ClassVar[str]       = (
        "KG-grounded ambiguity-aware diagnostic agent that clarifies "
        "underspecified symptom queries before downstream LLM generation. "
        "Uses POMDP belief state updated by Bayes rule to determine "
        "when to ask, answer, or abstain."
    )
    inputs:  ClassVar[List[str]]     = ["payload_json"]
    outputs: ClassVar[List[str]]     = ["json"]

    _driver:        Any              = PrivateAttr()
    _present:       Set[str]         = PrivateAttr(default_factory=set)
    _absent:        Set[str]         = PrivateAttr(default_factory=set)
    _asked:         List[str]        = PrivateAttr(default_factory=list)
    _candidates:    List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _question_log:  List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _belief:        Dict[str, float] = PrivateAttr(default_factory=dict)
    _llm_client:    Any              = PrivateAttr(default=None)
    _llm_cache:     Dict[str, Any]   = PrivateAttr(default_factory=dict)

    def __init__(self, uri=None, user=None, password=None,
                 llm_client=None, **kwargs):
        super().__init__(**kwargs)
        self._driver = GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            auth=(
                user or os.getenv("NEO4J_USER", "neo4j"),
                password or os.getenv("NEO4J_PASS", "12345678"),
            ),
        )
        # Optional LLM client for Tier 4 symptom resolution fallback.
        # Accepts any client with a .chat.completions.create() method
        # (OpenAI, AzureOpenAI, etc.).
        # When None, Tier 4 is skipped and unknown symptoms are dropped.
        self._llm_client = llm_client
        self._llm_cache  = {}

    # ─────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._present      = set()
        self._absent       = set()
        self._asked        = []
        self._candidates   = []
        self._question_log = []
        self._belief       = {}

    # ─────────────────────────────────────────────────────
    # KG lookup helpers (unchanged from original)
    # ─────────────────────────────────────────────────────

    def _name(self, cui: str) -> str:
        with self._driver.session() as s:
            row = s.run(CY_NAME, cui=cui, timeout=QUERY_TIMEOUT).single()
            return (row and row["name"]) or cui

    def _resolve_symptom(
        self, text: str, limit: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Three-tier symptom resolution pipeline.

        Tier 1: Exact normalized string match in KG (score 10.0)
        Tier 2: Synonym match in KG              (score  9.0)
        Tier 3: Full-text index search in KG     (score  8.x)
        Tier 4: LLM fallback → KG verification   (score  7.0)
                Only fires when llm_client is set and Tiers 1-3 fail.
                LLM maps plain-English symptom to UMLS concept name,
                then verifies the concept exists in the KG.
                Result is cached to avoid repeated API calls.

        The LLM is confined to lexical name resolution only.
        It never influences hypothesis construction or belief updates.
        """
        if not text:
            return []

        norm = _normalize(text)

        # ── Tier 1: exact normalized match ───────────────
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH (n)
                WHERE (n:Symptom OR n:Diagnosis)
                  AND toLower(n.name) = $n
                RETURN n.cui AS cui, n.name AS name, 10.0 AS score
                LIMIT $lim
                """,
                n=norm,
                lim=limit,
                timeout=QUERY_TIMEOUT,
            ).data()
        if rows:
            return [(r["cui"], r["name"], r["score"]) for r in rows]

        # ── Tier 2: synonym match ─────────────────────────
        with self._driver.session() as s:
            rows = s.run(
                """
                MATCH (n)
                WHERE (n:Symptom OR n:Diagnosis)
                  AND any(syn IN split(coalesce(n.synonyms, ''), '|')
                          WHERE trim(syn) <> ''
                            AND toLower(trim(syn)) = $n)
                RETURN n.cui AS cui, n.name AS name, 9.0 AS score
                LIMIT $lim
                """,
                n=norm,
                lim=limit,
                timeout=QUERY_TIMEOUT,
            ).data()
        if rows:
            return [(r["cui"], r["name"], r["score"]) for r in rows]

        # ── Tier 3: full-text index search ────────────────
        escaped = re.sub(
            r'([+\-!(){}$begin:math:display$$end:math:display$^"~*?:\\/]|&&|\|\|)',
            r'\\\1', norm
        )
        try:
            with self._driver.session() as s:
                rows = s.run(
                    f"""
                    CALL db.index.fulltext.queryNodes('{FT_INDEX}', $q)
                    YIELD node, score
                    WHERE (node:Symptom OR node:Diagnosis)
                    RETURN node.cui AS cui, node.name AS name, score
                    ORDER BY score DESC
                    LIMIT $lim
                    """,
                    q=f'"{escaped}"',
                    lim=limit,
                    timeout=QUERY_TIMEOUT,
                ).data()
            if rows:
                return [(r["cui"], r["name"], r["score"]) for r in rows]
        except Exception:
            pass

        # ── Tier 4: LLM fallback ──────────────────────────
        # Only fires when llm_client is set and Tiers 1-3 all failed.
        # LLM maps the plain-English symptom to a UMLS concept name,
        # then we verify that concept exists in the KG before using it.
        if self._llm_client is not None:
            result = self._llm_resolve_symptom(text, norm)
            if result:
                return [result]

        return []

    def _llm_resolve_symptom(
        self, original: str, normalized: str
    ) -> Optional[Tuple[str, str, float]]:
        """
        Tier 4: Use LLM to map an unknown symptom to a UMLS concept,
        then verify the concept exists in the KG.

        The LLM is given the symptom name and a sample of KG concept
        names to ground its response. It returns a concept name which
        we then look up directly in the KG.

        Results are cached in _llm_cache so each symptom is
        resolved at most once per session regardless of how many
        patients mention it.
        """
        cache_key = normalized
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        # Sample KG symptom names to ground the LLM
        try:
            with self._driver.session() as s:
                sample_rows = s.run(
                    """
                    MATCH (n:Symptom)
                    RETURN n.name AS name
                    ORDER BY rand()
                    LIMIT 40
                    """,
                    timeout=QUERY_TIMEOUT,
                ).data()
            sample_names = [r["name"] for r in sample_rows]
        except Exception:
            sample_names = []

        sample_text = "\n".join(f"- {n}" for n in sample_names)

        prompt = (
            f"You are a medical terminology expert.\n"
            f"Map this patient symptom to the closest UMLS concept name "
            f"in a medical knowledge graph.\n\n"
            f"Patient symptom: \"{original}\"\n\n"
            f"Sample concept names from the knowledge graph:\n"
            f"{sample_text}\n\n"
            f"Rules:\n"
            f"1. Return ONLY the concept name — no explanation\n"
            f"2. If the symptom matches a sample name exactly, return that\n"
            f"3. Otherwise return the closest medical equivalent\n"
            f"4. Keep it under 6 words\n\n"
            f"Concept name:"
        )

        try:
            resp = self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            kg_name = resp.choices[0].message.content.strip()

            # Verify the returned name exists in the KG
            with self._driver.session() as s:
                rows = s.run(
                    """
                    MATCH (n)
                    WHERE (n:Symptom OR n:Diagnosis)
                      AND toLower(n.name) = toLower($name)
                    RETURN n.cui AS cui, n.name AS name
                    LIMIT 1
                    """,
                    name=kg_name,
                    timeout=QUERY_TIMEOUT,
                ).data()

            if rows:
                result = (rows[0]["cui"], rows[0]["name"], 7.0)
                self._llm_cache[cache_key] = result
                return result

        except Exception:
            pass

        # Cache negative result to avoid repeated failed calls
        self._llm_cache[cache_key] = None
        return None

    def _search_symptom(
        self, text: str, limit: int = 5
    ) -> List[Tuple[str, str, float]]:
        return self._resolve_symptom(text, limit)

    def _build_candidates(self) -> List[Dict[str, Any]]:
        present = list(self._present)
        if not present:
            return []

        min_match = min(MIN_MATCH_CAP, len(present))

        with self._driver.session() as s:
            rows = s.run(
                CY_CANDIDATES,
                symptom_cuis=present,
                min_match=min_match,
                topk=TOPK_DIAG,
                timeout=QUERY_TIMEOUT,
            ).data()

        out: List[Dict[str, Any]] = []
        for r in rows:
            name = r["name"]
            if _is_bad_diagnosis_name(name):
                continue

            k           = int(r["k"])
            total       = int(r["total"])
            match_ratio = float(r["match_ratio"])
            score       = (2.0 * k) + match_ratio - (0.01 * total)

            out.append({
                "cui":         r["cui"],
                "name":        name,
                "k":           k,
                "total":       total,
                "all_sym":     set(r["all_sym"]),
                "match_ratio": match_ratio,
                "score":       score,
            })

        return sorted(out, key=lambda x: x["score"], reverse=True)

    def _refresh(self) -> None:
        self._candidates = self._build_candidates()
        if self._absent and HARD_FILTER_ABSENT and self._candidates:
            filtered = [
                c for c in self._candidates
                if not (c["all_sym"] & self._absent)
            ]
            if filtered:
                self._candidates = filtered

    def _refresh_triage(self) -> None:
        self._refresh()

    def _select_questions(
        self,
        epsilon: float = DEFAULT_EPSILON,
        max_k:   int   = MAX_QUESTIONS,
    ) -> Tuple[List[str], float]:
        if not self._candidates or max_k <= 0:
            return [], 0.0

        known   = self._present | self._absent | set(self._asked)
        dx_cuis = [c["cui"] for c in self._candidates]

        with self._driver.session() as s:
            rows = s.run(
                CY_DISTINGUISHING,
                dx_cuis=dx_cuis,
                known=list(known),
                limit=80,
                timeout=QUERY_TIMEOUT,
            ).data()

        filtered_rows = [
            r for r in rows if not _is_bad_question_symptom(r["name"])
        ]
        pool = [r["cui"] for r in filtered_rows if r["cui"] not in known]

        if not pool:
            return [], 0.0

        scored = [
            (ig, cui) for cui in pool
            if (ig := _ig(self._candidates, cui)) > 0
        ]
        scored.sort(reverse=True)

        best_ig  = scored[0][0] if scored else 0.0
        top_cuis = [cui for ig, cui in scored[:max_k] if ig >= epsilon]
        return top_cuis, best_ig

    def _best_next_symptoms(self, max_k: int = MAX_QUESTIONS) -> List[str]:
        cuis, _ = self._select_questions(max_k=max_k)
        return cuis

    def _rephrase_question(
        self, symptom_name: str, openai_client: Any = None
    ) -> str:
        if openai_client is None:
            return f"Do you also have {symptom_name}?"
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rephrase the medical symptom '{symptom_name}' "
                        "as a simple yes/no question for a patient. "
                        "Return only the question. Under 15 words."
                    )
                }],
                temperature=0,
                max_tokens=40,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"Do you also have {symptom_name}?"

    # ─────────────────────────────────────────────────────
    # Legacy stopping (kept for _execute backward compat)
    # ─────────────────────────────────────────────────────

    def _should_stop(
        self, epsilon: float
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._candidates:
            return True, "no_cand", {}
        if len(self._candidates) == 1:
            return True, "single", self._candidates[0]
        _, best_ig = self._select_questions(epsilon=0.0, max_k=1)
        if best_ig < epsilon:
            return True, "low_ig", self._candidates[0]
        return False, "", {}

    def _confidence_done(
        self, ratio: float
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self._candidates:
            return False, {}
        top = self._candidates[0]
        if len(self._candidates) == 1:
            return (top["k"] >= 1), top if top["k"] >= 1 else {}
        second = self._candidates[1]
        if top["k"] < 1:
            return False, {}
        if top["score"] >= ratio * second["score"]:
            return True, top
        return False, {}

    # ─────────────────────────────────────────────────────
    # NEW: POMDP stopping (Howard 1966)
    # ─────────────────────────────────────────────────────

    def _should_stop_pomdp(
        self, delta: float = DELTA
    ) -> Tuple[bool, str]:
        """
        Two stopping conditions grounded in decision theory.

        Condition 1 (belief concentration):
            Stop when max(b) > 1 - delta.
            delta = 0.05 → agent acts when P(top dx) > 0.95.
            Grounded in Howard (1966) Value of Information:
            under zero elicitation cost, act when posterior
            risk falls below the clinical significance threshold.

        Condition 2 (zero information gain):
            Stop when IG(best variable) = 0.
            No remaining question can reduce uncertainty.
            Grounded in the definition of expected information gain.

        Returns (should_stop, reason) where reason is one of:
            'belief_concentrated' | 'zero_ig' | 'no_candidates'
        """
        if not self._candidates:
            return True, "no_candidates"

        if len(self._candidates) == 1:
            return True, "belief_concentrated"

        # Condition 1
        if _max_belief(self._belief) > (1.0 - delta):
            return True, "belief_concentrated"

        # Condition 2
        _, best_ig = self._select_questions(epsilon=0.0, max_k=1)
        if best_ig <= 0.0:
            return True, "zero_ig"

        return False, ""

    # ─────────────────────────────────────────────────────
    # NEW: IS_A family assignment
    # ─────────────────────────────────────────────────────

    def _get_family(self, dx_cui: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (family_cui, family_name) for a diagnosis CUI via IS_A."""
        try:
            with self._driver.session() as s:
                rows = s.run(
                    CY_FAMILY,
                    cui=dx_cui,
                    timeout=QUERY_TIMEOUT,
                ).data()
            if rows:
                # Return the most specific family (first result)
                return rows[0]["cui"], rows[0]["name"]
        except Exception:
            pass
        return None, None

    def _get_shared_family(
        self, dx_cuis: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (family_cui, family_name) for the IS_A family that covers
        the most candidates, requiring at least ceil(n/2) coverage.

        Uses majority threshold rather than requiring all candidates
        to share a parent. This handles clinically related groups where
        one candidate (e.g. 'Sinusitis') is itself an ancestor of others
        (e.g. 'Viral sinusitis'), causing asymmetric IS_A coverage.

        Majority threshold: ceil(n_candidates / 2)
        For n=3: needs 2.  For n=4: needs 2.  For n=5: needs 3.
        """
        if not dx_cuis:
            return None, None

        import math
        min_coverage = math.ceil(len(dx_cuis) / 2)

        try:
            with self._driver.session() as s:
                row = s.run(
                    CY_SHARED_FAMILY,
                    dx_cuis=dx_cuis,
                    min_coverage=min_coverage,
                    timeout=QUERY_TIMEOUT,
                ).single()
            if row:
                return row["family_cui"], row["family_name"]
        except Exception:
            pass
        return None, None

    # ─────────────────────────────────────────────────────
    # NEW: Agent state assignment
    # ─────────────────────────────────────────────────────

    def _assign_agent_state(
        self,
        stop_reason: str,
        delta: float = DELTA,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Assign agent state based on stopping reason and belief state.

        Returns (state, family_cui, family_name) where state is:
            'ANSWER'  — max(b) > 1-delta, act on top hypothesis
            'PARTIAL' — IG=0 but all candidates share IS_A family,
                        actionable at family level
            'ABSTAIN' — IG=0 and candidates span multiple families,
                        cannot act safely

        Clinical interpretation:
            ANSWER  → pass clarified query to LLM for final prediction
            PARTIAL → pass to LLM with family constraint
            ABSTAIN → do not call LLM; return structured abstention
        """
        if stop_reason == "belief_concentrated" or stop_reason == "single":
            top_cui = _top_belief_cui(self._belief)
            if top_cui:
                fam_cui, fam_name = self._get_family(top_cui)
                return "ANSWER", fam_cui, fam_name
            return "ANSWER", None, None

        # IG = 0 or no candidates — check for family consensus
        if self._candidates:
            dx_cuis = [c["cui"] for c in self._candidates]
            fam_cui, fam_name = self._get_shared_family(dx_cuis)
            if fam_cui:
                return "PARTIAL", fam_cui, fam_name

        return "ABSTAIN", None, None

    # ─────────────────────────────────────────────────────
    # _execute (openCHA interface — unchanged)
    # ─────────────────────────────────────────────────────

    def _explain(
        self, stop: bool, reason: str, top: Dict[str, Any]
    ) -> str:
        if stop and top:
            return (
                f"Most likely: '{top['name']}' "
                f"(matched {top['k']}/{top['total']} symptoms). "
                f"Stop: {reason}."
            )
        if not self._candidates:
            return "No candidates found for these symptoms."
        names = ", ".join(c["name"] for c in self._candidates[:3])
        return f"Current differentials: {names}."

    def _payload(
        self,
        next_sx: List[str],
        stop: bool,
        reason: str,
        top: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "done":         bool(stop),
            "stop_reason":  reason,
            "n_hypotheses": len(self._candidates),
            "entropy":      round(_uniform_entropy(len(self._candidates)), 4),
            "top_candidate": (
                top.get("name", "")
                if top
                else (self._candidates[0]["name"] if self._candidates else "")
            ),
            "explanation":   self._explain(stop, reason, top),
            "candidates":    [
                {
                    "cui":   c["cui"],
                    "name":  c["name"],
                    "score": round(c["score"], 4),
                    "match": f"{c['k']}/{c['total']}",
                }
                for c in self._candidates[:10]
            ],
            "next_questions": [
                {"cui": s, "name": self._name(s)} for s in (next_sx or [])
            ],
            "clarified_symptoms": (
                [
                    {"cui": c, "name": self._name(c), "status": "present"}
                    for c in self._present
                ] + [
                    {"cui": c, "name": self._name(c), "status": "absent"}
                    for c in self._absent
                ]
            ),
            "question_log": list(self._question_log),
        }

    def _execute(self, inputs: List[str]) -> str:
        """openCHA interface — unchanged from original."""
        t0 = time.time()
        try:
            payload = (
                json.loads(inputs[0]) if inputs and inputs[0] else {}
            )
        except Exception:
            payload = {}

        qtype     = str(payload.get("query_type", "")).lower()
        texts     = payload.get("symptoms_text", []) or []
        cui       = str(payload.get("cui", "")).strip()
        present   = payload.get("present", None)
        epsilon   = float(payload.get("epsilon", DEFAULT_EPSILON))
        max_q     = int(payload.get("max_next_questions", MAX_QUESTIONS))
        conf_ratio = float(payload.get("confidence_ratio", 1.2))

        try:
            if qtype == "triage_reset":
                self._reset()
                result = self._payload([], False, "reset", {})

            elif qtype == "triage_start":
                self._reset()
                for tx in texts:
                    hits = self._resolve_symptom(tx)
                    if hits:
                        self._present.add(hits[0][0])

                self._refresh()
                stop, reason, top = self._should_stop(epsilon)

                if not stop:
                    confident, top_conf = self._confidence_done(conf_ratio)
                    if confident:
                        stop, reason, top = True, "confidence", top_conf

                nxt, _ = (
                    self._select_questions(epsilon, max_q)
                    if not stop else ([], 0.0)
                )
                self._asked.extend(nxt)
                result = self._payload(nxt, stop, reason, top)

            elif qtype == "triage_answer":
                if cui:
                    if bool(present):
                        self._present.add(cui)
                    else:
                        self._absent.add(cui)

                    self._question_log.append({
                        "cui":     cui,
                        "name":    self._name(cui),
                        "present": bool(present),
                    })

                self._refresh()
                stop, reason, top = self._should_stop(epsilon)

                if not stop:
                    confident, top_conf = self._confidence_done(conf_ratio)
                    if confident:
                        stop, reason, top = True, "confidence", top_conf

                nxt, _ = (
                    self._select_questions(epsilon, max_q)
                    if not stop else ([], 0.0)
                )
                nxt    = [s for s in nxt if s not in self._asked]
                self._asked.extend(nxt)
                result = self._payload(nxt, stop, reason, top)

            elif qtype == "triage_status":
                stop, reason, top = self._should_stop(epsilon)

                if not stop:
                    confident, top_conf = self._confidence_done(conf_ratio)
                    if confident:
                        stop, reason, top = True, "confidence", top_conf

                nxt, _ = (
                    self._select_questions(epsilon, max_q)
                    if not stop else ([], 0.0)
                )
                nxt    = [s for s in nxt if s not in self._asked]
                self._asked.extend(nxt)
                result = self._payload(nxt, stop, reason, top)

            else:
                return json.dumps({
                    "ok":    False,
                    "error": f"unknown query_type: {qtype}",
                })

            return json.dumps({
                "ok":          True,
                "query_type":  qtype,
                "result":      result,
                "diagnostics": {
                    "elapsed_ms": int((time.time() - t0) * 1000)
                },
            })

        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    # ─────────────────────────────────────────────────────
    # NEW: POMDP benchmark runner
    # ─────────────────────────────────────────────────────

    def run_benchmark_case(
        self,
        remaining_symptoms: List[str],
        full_symptoms:      List[str],
        conversation:       str,
        ground_truth:       str,
        delta:              float = DELTA,
        openai_client:      Any   = None,
        model:              str   = "gpt-4o",
        rephrase:           bool  = False,
        # Legacy params kept for call-site backward compatibility
        # but not used in the POMDP logic
        epsilon:            float = DEFAULT_EPSILON,
        t_max:              int   = DEFAULT_T_MAX,
    ) -> Dict[str, Any]:
        """
        POMDP ambiguity-mitigation loop.

        Stopping is governed by two parameter-free conditions:

        1. Belief concentration: max(b) > 1 - delta
           Agent acts when posterior probability of the top
           hypothesis exceeds the clinical significance threshold.
           delta = 0.05 following Howard (1966) VOI framework.

        2. Zero information gain: IG(best variable) = 0
           No remaining question can reduce hypothesis uncertainty.

        Agent state on exit:
            ANSWER  — belief concentrated, pass to LLM
            PARTIAL — IG=0, family consensus found, pass to LLM
            ABSTAIN — IG=0, no family consensus, skip LLM
        """
        self._reset()

        # ── Resolve initial visible symptoms ──────────────
        resolved_initial_symptoms: List[str] = []
        for sx in remaining_symptoms:
            hits = self._resolve_symptom(sx, limit=3)
            if hits:
                self._present.add(hits[0][0])
                resolved_initial_symptoms.append(hits[0][1])

        self._refresh()

        # ── Initial state ─────────────────────────────────
        n_init        = len(self._candidates)
        entropy_init  = _uniform_entropy(n_init)
        candidates_init = [
            {"name": c["name"], "score": round(c["score"], 4)}
            for c in self._candidates[:10]
        ]

        # Initialize POMDP belief state
        self._belief = _belief_init(self._candidates)

        # ── Resolve oracle (full symptom) CUIs ───────────
        full_symptom_cuis: Set[str] = set()
        for sx in full_symptoms:
            hits = self._resolve_symptom(sx, limit=3)
            if hits:
                full_symptom_cuis.add(hits[0][0])

        # ── POMDP clarification loop ───────────────────────
        turns       = 0
        stop_reason = "single"

        while True:
            should_stop, reason = self._should_stop_pomdp(delta)
            if should_stop:
                stop_reason = reason
                break

            # Select max-IG question
            nxt, best_ig = self._select_questions(epsilon=0.0, max_k=1)
            if not nxt or best_ig <= 0.0:
                stop_reason = "zero_ig"
                break

            ask_cui  = nxt[0]
            ask_name = self._name(ask_cui)

            if rephrase and openai_client is not None:
                question_text = self._rephrase_question(
                    ask_name, openai_client
                )
            else:
                question_text = f"Do you also have {ask_name}?"

            # Oracle answer
            answer = ask_cui in full_symptom_cuis

            # Update hard sets
            if answer:
                self._present.add(ask_cui)
            else:
                self._absent.add(ask_cui)

            # Update POMDP belief by Bayes rule
            self._belief = _belief_update(
                self._belief, self._candidates, ask_cui, answer
            )

            self._question_log.append({
                "cui":      ask_cui,
                "name":     ask_name,
                "question": question_text,
                "present":  answer,
                "ig":       round(best_ig, 4),
                "max_b":    round(_max_belief(self._belief), 4),
            })
            self._asked.append(ask_cui)
            self._refresh()

            # Re-initialize belief for survivors after hard filter
            # (candidates may have been pruned by HARD_FILTER_ABSENT)
            surviving_cuis = {c["cui"] for c in self._candidates}
            self._belief = {
                cui: p for cui, p in self._belief.items()
                if cui in surviving_cuis
            }
            total = sum(self._belief.values())
            if total > 0:
                self._belief = {
                    cui: p / total
                    for cui, p in self._belief.items()
                }
            else:
                # Degenerate: re-initialize uniform
                self._belief = _belief_init(self._candidates)

            turns += 1

        # ── Assign agent state ────────────────────────────
        agent_state, family_cui, family_name = self._assign_agent_state(
            stop_reason, delta
        )

        # ── Final metrics ─────────────────────────────────
        entropy_final   = _uniform_entropy(len(self._candidates))
        candidates_final = [
            {"name": c["name"], "score": round(c["score"], 4)}
            for c in self._candidates[:10]
        ]
        max_belief_final = round(_max_belief(self._belief), 4)
        top_cui_final    = _top_belief_cui(self._belief)
        top_name_final   = (
            self._name(top_cui_final) if top_cui_final else ""
        )

        # ── Build clarified query (passed to LLM) ─────────
        clarif = ""
        if self._question_log:
            clarif = "\n\nClarification answers:\n"
            for q in self._question_log:
                ans     = "Yes" if q["present"] else "No"
                clarif += f"  - {q['name']}: {ans}\n"

        hyp = ""
        if self._candidates:
            names = [c["name"] for c in self._candidates[:5]]
            hyp   = f"\n\nKG candidate diagnoses: {', '.join(names)}."

        family_note = ""
        if family_name and agent_state in ("PARTIAL", "ANSWER"):
            family_note = f"\n\nClinical family: {family_name}."

        clarified_query = f"{conversation}{clarif}{hyp}{family_note}"

        # ── Abstained flag (backward compat) ──────────────
        abstained = (agent_state == "ABSTAIN")

        return {
            # ── Core outputs for LLM and evaluation ──
            "ground_truth":               ground_truth,
            "conversation":               conversation,
            "remaining_symptoms":         remaining_symptoms,
            "full_symptoms":              full_symptoms,
            "resolved_initial_symptoms":  resolved_initial_symptoms,
            "clarified_query":            clarified_query,

            # ── Agent state (new) ──
            "agent_state":    agent_state,   # ANSWER / PARTIAL / ABSTAIN
            "family_cui":     family_cui,
            "family_name":    family_name,
            "max_belief":     max_belief_final,
            "top_candidate":  top_name_final,

            # ── Hypothesis set metrics ──
            "n_hypotheses_init":   n_init,
            "n_hypotheses_final":  len(self._candidates),
            "entropy_init":        round(entropy_init,  4),
            "entropy_final":       round(entropy_final, 4),
            "candidates_init":     candidates_init,
            "candidates_final":    candidates_final,

            # ── Clarification metrics ──
            "turns":         turns,
            "stop_reason":   stop_reason,
            "abstained":     abstained,
            "question_log":  list(self._question_log),
        }


if __name__ == "__main__":
    task = DiagnosticAmbiguityTask(password="12345678")

    print("Testing POMDP triage with ['Cough', 'Fever', 'Sore Throat'] ...")
    result = task.run_benchmark_case(
        remaining_symptoms=["Cough", "Fever", "Sore Throat"],
        full_symptoms=["Cough", "Fever", "Sore Throat", "Swollen Tonsils"],
        conversation="Patient: I have cough, fever and sore throat.",
        ground_truth="Streptococcal sore throat",
        delta=0.05,
    )

    print(f"Agent state   : {result['agent_state']}")
    print(f"Family        : {result['family_name']}")
    print(f"Max belief    : {result['max_belief']}")
    print(f"Top candidate : {result['top_candidate']}")
    print(f"Turns         : {result['turns']}")
    print(f"Stop reason   : {result['stop_reason']}")
    print(f"Candidates    : {[c['name'] for c in result['candidates_final']]}")