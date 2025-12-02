# src/openCHA/tasks/ambiguity/diagnostic_ambiguity_task.py

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, ClassVar, Dict, List, Set, Tuple

from neo4j import GraphDatabase
from pydantic import PrivateAttr

from ..task import BaseTask


# =========================
# Config
# =========================

QUERY_TIMEOUT_SECS = 5.0

# Neo4j fulltext index over Symptom.name (and/or synonym fields)
FT_INDEX = "term_name_ft"

# How many diagnoses to keep when ranking
TOPK_DIAG = 25

# How many new symptoms to ask per step
MAX_NEXT_QUESTIONS = 10

# If True: drop diagnoses that require symptoms marked ABSENT
HARD_FILTER_ABSENT = True

# Require top score >= CONFIDENCE_RATIO * second score to stop
CONFIDENCE_RATIO = 1.2
# Require at least this many matched symptoms to consider a dx “supported”
MIN_EVIDENCE = 1

# Filter out diagnoses whose names start with "[" (often junk/technical)
FILTER_BRACKETED = True


# =========================
# Symptom normalization
# =========================

# Map messy phrases from dataset / conversation → canonical symptom names.
# Extend this dict using your CSV + KG + UMLS helper script.
NORMALIZE_SYMPTOMS: Dict[str, str] = {
    "stuffy nose": "nasal congestion",
    "runny nose": "nasal congestion",
    "runny/stuffy nose": "nasal congestion",
    "yellow mucus": "sputum production",
    "colored sputum": "sputum production",
    "short of breath": "dyspnea",
    "trouble breathing": "dyspnea",
    "burning pee": "dysuria",
    "pain when peeing": "dysuria",
    # add dataset- and conversation-specific variants here…
}


def _normalize_symptom_text(text: str) -> str:
    """
    Light normalization + manual synonym mapping so that symptom strings
    from CSV / conversation better align with KG node names.
    """
    if not text:
        return ""
    t = text.strip().lower()
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    return NORMALIZE_SYMPTOMS.get(t, t)


# =========================
# Cypher templates
# =========================

# Full-text search over Symptom nodes
CY_SEARCH_SYMPTOM = f"""
CALL db.index.fulltext.queryNodes('{FT_INDEX}', $q) YIELD node, score
WHERE node:Symptom
RETURN node.cui AS cui, node.name AS name, score
ORDER BY score DESC
LIMIT $limit
"""

# Name lookup by CUI (works for Symptom / Diagnosis if they have "cui" + "name")
CY_NAME = """
MATCH (c {cui: $cui})
RETURN coalesce(c.name, $cui) AS name
"""

# Candidate diagnoses given present symptom CUIs
CY_DIAG_CANDIDATES_BASE = """
UNWIND $symptom_cuis AS scui
MATCH (d:Diagnosis)-[:HAS_SYMPTOM]->(s:Symptom {cui: scui})
{WHERE_FILTERS}
WITH d, collect(DISTINCT s.cui) AS matched, size($symptom_cuis) AS reported
MATCH (d)-[:HAS_SYMPTOM]->(allS:Symptom)
WITH d, matched, reported, collect(DISTINCT allS.cui) AS all_sym
WHERE size(all_sym) >= 3
RETURN d.cui  AS cui,
       coalesce(d.name) AS name,
       size(matched)    AS k,
       size(all_sym)    AS total,
       all_sym
ORDER BY k DESC, total ASC
LIMIT $topk
"""

# Extra symptoms per diagnosis (used for fallback question selection)
CY_DIAG_EXTRA_SYMPTOMS = """
MATCH (d:Diagnosis {cui: $dcui})-[:HAS_SYMPTOM]->(s:Symptom)
WHERE NOT s.cui IN $known
RETURN s.cui AS cui, coalesce(s.name, s.cui) AS name
LIMIT $limit
"""


# =========================
# Utility functions
# =========================

def _entropy(p: float) -> float:
    """Binary entropy H(p). Used for information gain."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _where_filters() -> str:
    """
    Filter out low-quality diagnoses from the KG:
    bracketed labels, NOS / unspecified, etc.
    """
    clauses: List[str] = []

    if FILTER_BRACKETED:
        clauses.append("WHERE NOT d.name STARTS WITH '['")

    bad_bits = [
        "adverse",
        "classified elsewhere",
        "nos",
        "other specified",
        "unspecified",
        "in diseases classified elsewhere",
    ]
    for bit in bad_bits:
        prefix = "WHERE" if not clauses else "AND"
        clauses.append(f"{prefix} NOT toLower(d.name) CONTAINS '{bit}'")

    return "\n".join(clauses)


# =========================
# Main Task
# =========================

class DiagnosticAmbiguityTask(BaseTask):
    """
    Ambiguity-aware diagnostic agent over a Neo4j Diagnosis–Symptom KG.

    It:
      • maps free-text symptom strings → Symptom CUIs via full-text search
      • ranks candidate diagnoses using KG overlap with PRESENT symptoms
      • chooses distinguishing symptoms (high information gain) as next questions
      • updates candidates with yes/no (present/absent) answers
      • stops when confident and returns top differentials
      • exposes clarified_symptoms and question_log for LLM / analysis

    query_type options in payload_json:
      - triage_start  : start a new case with initial symptoms_text
      - triage_answer : update with answer for a single symptom CUI
      - triage_status : inspect current status + propose more questions
      - triage_reset  : reset internal state
    """

    name: ClassVar[str] = "diagnostic_ambiguity"
    chat_name: ClassVar[str] = "DiagnosticAmbiguityAgent"
    description: ClassVar[str] = """
    Ambiguity-aware diagnostic agent over a Neo4j Diagnosis–Symptom KG.
    Use triage_start / triage_answer / triage_status / triage_reset.
    """
    inputs: ClassVar[List[str]] = ["payload_json"]
    outputs: ClassVar[List[str]] = ["json"]

    # runtime state (per agent instance)
    _driver: Any = PrivateAttr()

    _present: Set[str] = PrivateAttr(default_factory=set)        # symptom CUIs marked present
    _absent: Set[str] = PrivateAttr(default_factory=set)         # symptom CUIs marked absent
    _asked: List[str] = PrivateAttr(default_factory=list)        # symptom CUIs already asked
    _candidates: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _question_log: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        uri = uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASS", "12345678")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    # =========================
    # Parsing / name helpers
    # =========================

    def _parse_inputs(
        self,
        inputs: List[str],
    ) -> Tuple[str, List[str], str, Any, float, int]:
        """
        Expected payload_json structure:

        {
          "query_type": "triage_start" | "triage_answer" | "triage_status" | "triage_reset",
          "symptoms_text": [...],       # only for triage_start
          "cui": "SOME_SYMPTOM_CUI",    # only for triage_answer
          "present": true/false,        # only for triage_answer
          "confidence_ratio": float,
          "max_next_questions": int
        }
        """
        payload: Dict[str, Any] = {}
        if inputs and inputs[0]:
            try:
                payload = json.loads(inputs[0])
            except Exception:
                payload = {}

        qtype = str(payload.get("query_type", "")).lower()
        texts = payload.get("symptoms_text", []) or []
        cui = str(payload.get("cui", "")).strip()
        present_flag = payload.get("present", None)
        conf_ratio = float(payload.get("confidence_ratio", CONFIDENCE_RATIO))
        max_next = int(payload.get("max_next_questions", MAX_NEXT_QUESTIONS))

        return qtype, texts, cui, present_flag, conf_ratio, max_next

    def _name(self, cui: str) -> str:
        """Resolve a CUI into a displayable name using the KG."""
        with self._driver.session() as s:
            row = s.run(CY_NAME, cui=cui, timeout=QUERY_TIMEOUT_SECS).single()
            return (row and row["name"]) or cui

    # =========================
    # Symptom / candidate retrieval
    # =========================

    def _search_symptom(self, text: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        """
        Map free-text symptom string to top-k Symptom nodes using the full-text index.

        Steps:
          1) normalize text with _normalize_symptom_text
          2) escape Lucene special chars
          3) wrap in quotes → phrase query for queryNodes
        """
        if not text or not text.strip():
            return []

        raw = _normalize_symptom_text(text)
        if not raw:
            return []

        # Escape Lucene special characters:
        # + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        lucene_special = r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)'
        escaped = re.sub(lucene_special, r'\\\1', raw)

        q = f'"{escaped}"'

        with self._driver.session() as s:
            rows = s.run(
                CY_SEARCH_SYMPTOM,
                q=q,
                limit=limit,
                timeout=QUERY_TIMEOUT_SECS,
            ).data()

        return [(r["cui"], r["name"], r["score"]) for r in rows]

    def _candidates_from_present(
        self,
        present: List[str],
        topk: int = TOPK_DIAG,
    ) -> List[Dict[str, Any]]:
        """Compute candidate diagnoses given a list of PRESENT symptom CUIs."""
        if not present:
            return []

        cy = CY_DIAG_CANDIDATES_BASE.replace("{WHERE_FILTERS}", _where_filters())
        with self._driver.session() as s:
            rows = s.run(
                cy,
                symptom_cuis=present,
                topk=topk,
                timeout=QUERY_TIMEOUT_SECS,
            ).data()

        out: List[Dict[str, Any]] = []
        for r in rows:
            cui = r["cui"]
            name = r["name"]
            k = r["k"]
            total = r["total"]
            all_sym = set(r["all_sym"])

            # Jaccard-like score: matches / (present + total - matches)
            score = (k + 1e-6) / (len(present) + total - k + 1e-6)

            out.append(
                {
                    "cui": cui,
                    "name": name,
                    "k": k,
                    "total": total,
                    "all_sym": all_sym,
                    "score": score,
                }
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    # =========================
    # Core triage mechanics
    # =========================

    def _refresh_triage(self, hard_filter: bool = HARD_FILTER_ABSENT) -> None:
        """
        Recompute diagnosis candidates given current PRESENT / ABSENT sets,
        and apply basic filtering / penalty rules.
        """
        self._candidates = self._candidates_from_present(list(self._present))
        if not self._candidates:
            return

        # 1) Filter or penalize diagnoses that require ABSENT symptoms
        if self._absent:
            filtered: List[Dict[str, Any]] = []
            for c in self._candidates:
                conflict = c["all_sym"] & self._absent
                if hard_filter and conflict:
                    # drop diagnoses that contradict known ABSENT symptoms
                    continue
                penalty = 1.0 / (1.0 + len(conflict)) if conflict else 1.0
                c2 = dict(c)
                c2["score"] = c["score"] * penalty
                filtered.append(c2)
            self._candidates = sorted(filtered, key=lambda x: x["score"], reverse=True)

        # 2) Drop trivial diagnoses that add no new symptoms beyond PRESENT
        present_set = set(self._present)
        nontrivial = [c for c in self._candidates if (set(c["all_sym"]) - present_set)]
        if nontrivial:
            self._candidates = nontrivial

        # 3) If everything got filtered out, fall back to smaller candidate set
        if not self._candidates:
            self._candidates = self._candidates_from_present(list(self._present))[:10]

    def _fallback_next_symptoms(self, max_k: int) -> List[str]:
        """
        Conservative fallback: pull extra symptoms from top few diagnoses
        that are not yet PRESENT/ABSENT/ASKED.
        """
        if not self._candidates or max_k <= 0:
            return []

        known = self._present | self._absent | set(self._asked)
        wanted: List[str] = []
        seen: Set[str] = set()

        with self._driver.session() as s:
            for c in self._candidates[:3]:
                rows = s.run(
                    CY_DIAG_EXTRA_SYMPTOMS,
                    dcui=c["cui"],
                    known=list(known),
                    limit=max_k * 3,
                    timeout=QUERY_TIMEOUT_SECS,
                ).data()
                for r in rows:
                    scui = r["cui"]
                    if scui not in seen:
                        seen.add(scui)
                        wanted.append(scui)
                    if len(wanted) >= max_k:
                        break
                if len(wanted) >= max_k:
                    break

        return wanted

    def _best_next_symptoms(self, max_k: int = MAX_NEXT_QUESTIONS) -> List[str]:
        """
        Choose next symptoms by information gain:

        - build pool of all symptoms that appear in candidate diagnoses
        - remove symptoms we already know (PRESENT / ABSENT / ASKED)
        - score each candidate symptom using entropy over diagnoses
        - return top-k symptom CUIs
        """
        if not self._candidates or max_k <= 0:
            return []

        known = self._present | self._absent | set(self._asked)
        pool = set().union(*[c["all_sym"] for c in self._candidates]) - known
        if not pool:
            return self._fallback_next_symptoms(max_k)

        n = len(self._candidates)
        scored: List[Tuple[float, str]] = []

        for scui in pool:
            cnt = sum(1 for c in self._candidates if scui in c["all_sym"])
            ent = _entropy(cnt / n)
            scored.append((ent, scui))

        scored.sort(reverse=True)
        return [scui for _, scui in scored[:max_k]]

    def _confidence_done(self, ratio: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Decide whether we are confident enough to stop:

        - at least MIN_EVIDENCE matched symptoms, and
        - top score >= ratio * second score (if more than one candidate)
        """
        if not self._candidates:
            return False, {}

        top = self._candidates[0]

        if len(self._candidates) == 1:
            if top["k"] >= MIN_EVIDENCE:
                return True, top
            return False, {}

        second = self._candidates[1]

        if top["k"] < MIN_EVIDENCE:
            return False, {}

        if top["score"] >= ratio * second["score"]:
            return True, top

        return False, {}

    # =========================
    # Explanation / payload
    # =========================

    def _explain(self, done: bool, top: Dict[str, Any]) -> str:
        """Human-friendly explanation string."""
        if done and top:
            return (
                "Based on the clarified symptoms, the most likely diagnosis "
                f"is '{top['name']}' (matches {top['k']} of {top['total']} "
                "graph-linked symptoms)."
            )

        if not self._candidates:
            return "No diagnosis candidates were found for the current symptoms."

        c = self._candidates[:3]
        names = ", ".join(ci["name"] for ci in c)
        return f"Current leading differentials: {names}."

    def _clarified_symptoms_payload(self) -> List[Dict[str, Any]]:
        """
        Return a list of symptoms with status (present/absent),
        so the LLM can see a clean, de-ambiguous symptom set.
        """
        out: List[Dict[str, Any]] = []
        for scui in self._present:
            out.append({"cui": scui, "name": self._name(scui), "status": "present"})
        for scui in self._absent:
            out.append({"cui": scui, "name": self._name(scui), "status": "absent"})
        return out

    def _triage_payload(
        self,
        next_symptoms: List[str],
        done: bool,
        top: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Pack all relevant state into a JSON-serializable dict."""
        return {
            "done": bool(done),
            "explanation": self._explain(done, top),
            "candidates": [
                {
                    "cui": c["cui"],
                    "name": c["name"],
                    "score": round(c["score"], 4),
                    "match": f"{c['k']}/{c['total']}",
                }
                for c in self._candidates[:10]
            ],
            "next_questions": [
                {"cui": s, "name": self._name(s)} for s in (next_symptoms or [])
            ],
            "clarified_symptoms": self._clarified_symptoms_payload(),
            "question_log": list(self._question_log),
        }

    # =========================
    # Core entrypoint
    # =========================

    def _execute(self, inputs: List[str]) -> str:
        """
        Main entrypoint required by BaseTask.
        """
        t0 = time.time()
        qtype, texts, cui, present_flag, conf_ratio, max_next = self._parse_inputs(inputs)

        try:
            # ---------- reset ----------
            if qtype == "triage_reset":
                self._present = set()
                self._absent = set()
                self._asked = []
                self._candidates = []
                self._question_log = []
                result = self._triage_payload([], False, {})

            # ---------- start ----------
            elif qtype == "triage_start":
                self._present = set()
                self._absent = set()
                self._asked = []
                self._candidates = []
                self._question_log = []

                # Map free-text → Symptom CUIs (take top hit per text)
                for ttxt in texts or []:
                    hits = self._search_symptom(ttxt, limit=5)
                    if hits:
                        self._present.add(hits[0][0])

                self._refresh_triage()
                done, top = self._confidence_done(conf_ratio)
                next_sym = [] if done else self._best_next_symptoms(max_next)
                self._asked.extend(next_sym)

                result = self._triage_payload(next_sym, done, top)

            # ---------- answer ----------
            elif qtype == "triage_answer":
                if cui:
                    ans = bool(present_flag)
                    if ans:
                        self._present.add(cui)
                    else:
                        self._absent.add(cui)

                    # log question + answer
                    self._question_log.append(
                        {"cui": cui, "name": self._name(cui), "present": ans}
                    )

                self._refresh_triage()
                done, top = self._confidence_done(conf_ratio)
                next_sym = [] if done else self._best_next_symptoms(max_next)
                next_sym = [s for s in next_sym if s not in self._asked]
                self._asked.extend(next_sym)

                result = self._triage_payload(next_sym, done, top)

            # ---------- status ----------
            elif qtype == "triage_status":
                done, top = self._confidence_done(conf_ratio)
                next_sym = [] if done else self._best_next_symptoms(max_next)
                next_sym = [s for s in next_sym if s not in self._asked]
                self._asked.extend(next_sym)

                result = self._triage_payload(next_sym, done, top)

            else:
                return json.dumps(
                    {
                        "ok": False,
                        "error": (
                            f"unknown query_type: {qtype}. "
                            "Use triage_start / triage_answer / triage_status / triage_reset."
                        ),
                    }
                )

            elapsed_ms = int((time.time() - t0) * 1000)
            return json.dumps(
                {
                    "ok": True,
                    "query_type": qtype,
                    "params": {
                        "symptoms_text": texts,
                        "cui": cui,
                        "present": present_flag,
                        "confidence_ratio": conf_ratio,
                        "max_next_questions": max_next,
                    },
                    "result": result,
                    "diagnostics": {"elapsed_ms": elapsed_ms},
                }
            )

        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


# Quick manual test when running as a module
if __name__ == "__main__":
    task = DiagnosticAmbiguityTask()
    payload = {
        "query_type": "triage_start",
        "symptoms_text": ["fever", "cough", "chest pain"],
    }
    out = task._execute([json.dumps(payload)])
    print(out)