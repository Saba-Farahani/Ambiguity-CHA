# src/openCHA/tasks/ambiguity/food_ambiguity_task.py
"""
Food Safety Ambiguity Task — POMDP agent for dietary safety queries.

Architecture mirrors DiagnosticAmbiguityTask:
  Symptom task: symptoms known → diagnosis unknown → ask about symptoms
  Food task:    food known     → condition unknown → ask about conditions

POMDP:
  State:     patient's health conditions (latent)
  Hypotheses H: conditions the food is RISKY_FOR (from KG)
  Questions: "Do you have [condition]?"
  Oracle:    patient_context column (ground-truth condition)
  Decision:  NOT OKAY (patient has a risky condition)
             OKAY     (no risky condition found)
             PARTIAL  (family-level resolution via IS_A)
             ABSTAIN  (cannot resolve safely)

Stopping conditions (parameter-free, mirrors Algorithm 1):
  Condition 1: max(b) > 1 - delta  → ANSWER
  Condition 2: IG(v*) = 0          → PARTIAL or ABSTAIN

All clinical reasoning is deterministic (symbolic layer).
GPT-4o is called only for final response generation.
"""

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


# ─────────────────────────────────────────────────────────────────
# Constants — mirrors DiagnosticAmbiguityTask
# ─────────────────────────────────────────────────────────────────

TOPK_CONDITIONS  = 12       # retrieve top-12 from KG (broader coverage)
TOPK_ASK         = 10       # ask about top-10 in POMDP (9 NO answers → OKAY)
DELTA            = 0.05     # Howard (1966) VOI threshold: act when P > 1-δ
MAX_TURNS        = 10       # max clarification turns
DEFAULT_EPSILON = 0.01
DEFAULT_T_MAX   = 5


# ─────────────────────────────────────────────────────────────────
# Condition surface normalization
# Maps dataset patient_context strings → canonical condition names
# ─────────────────────────────────────────────────────────────────

CONDITION_NORMALIZE: Dict[str, str] = {
    "diabetes":                              "type 2 diabetes mellitus",
    "type 2 diabetes":                       "type 2 diabetes mellitus",
    "type 1 diabetes":                       "type 1 diabetes mellitus",
    "gestational diabetes":                  "gestational diabetes mellitus",
    "pre-diabetes":                          "prediabetes",
    "pre diabetes":                          "prediabetes",
    "acid reflux":                           "gastroesophageal reflux disease",
    "gerd":                                  "gastroesophageal reflux disease",
    "reflux":                                "gastroesophageal reflux disease",
    "ibs":                                   "irritable bowel syndrome",
    "high blood pressure":                   "hypertension",
    "high cholesterol":                      "hypercholesterolemia",
    "kidney disease":                        "chronic kidney disease",
    "ckd":                                   "chronic kidney disease",
    "heart disease":                         "coronary artery disease",
    "cardiovascular disease":               "coronary artery disease",
    "heart failure":                         "congestive heart failure",
    "chf":                                   "congestive heart failure",
    "afib":                                  "atrial fibrillation",
    "liver disease":                         "liver disease",
    "gluten intolerance":                    "non-celiac gluten sensitivity",
    "celiac":                                "celiac disease",
    "lactose intolerance":                   "lactose intolerance",
    "peanut allergy":                        "peanut allergy",
    "tree nut allergy":                      "tree nut allergy",
    "shellfish allergy":                     "shellfish allergy",
    "hypothyroidism":                        "hypothyroidism",
    "hyperthyroidism":                       "hyperthyroidism",
    "gout":                                  "gout",
    "crohn s disease":                       "crohn's disease",
    "crohn's disease":                       "crohn's disease",
    "ulcerative colitis":                    "ulcerative colitis",
    "no restrictions":                       "no restrictions",
    "healthy":                               "no restrictions",
}


def _norm_condition(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip().lower())
    return CONDITION_NORMALIZE.get(t, t)


def _norm_food(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


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


# ─────────────────────────────────────────────────────────────────
# Belief state helpers (mirrors diagnostic task)
# ─────────────────────────────────────────────────────────────────

def _belief_init(conditions: List[Dict]) -> Dict[str, float]:
    """Uniform belief over hypothesis conditions."""
    n = len(conditions)
    if n == 0:
        return {}
    p = 1.0 / n
    return {c["cid"]: p for c in conditions}


def _belief_update(
    belief: Dict[str, float],
    conditions: List[Dict],
    asked_cid: str,
    answer: bool,
) -> Dict[str, float]:
    """
    Bayesian update after patient answers whether they have a condition.

    If answer=YES → patient has condition → food is RISKY → collapse
    If answer=NO  → remove condition from hypothesis set
    """
    if not belief:
        return belief

    new_belief = {}
    for c in conditions:
        cid = c["cid"]
        if cid not in belief:
            continue
        if cid == asked_cid:
            if answer:
                # Patient confirmed → collapse belief to this condition
                new_belief[cid] = 1.0
            # If NO → remove (don't add to new_belief)
        else:
            if not answer:
                # Patient denied asked_cid → redistribute its probability
                new_belief[cid] = belief[cid]
            else:
                # Patient confirmed asked_cid → all others get 0
                pass

    # Normalize
    total = sum(new_belief.values())
    if total > 0:
        return {k: v / total for k, v in new_belief.items()}
    return new_belief


def _ig_condition(conditions: List[Dict], asked_cid: str) -> float:
    """
    Information gain of asking about asked_cid.
    Mirrors _ig() from diagnostic task.

    h_yes = conditions patient would have if YES (only asked_cid)
    h_no  = conditions remaining if NO (all others)
    """
    n = len(conditions)
    if n <= 1:
        return 0.0

    h_yes = [c for c in conditions if c["cid"] == asked_cid]
    h_no  = [c for c in conditions if c["cid"] != asked_cid]

    p_yes = len(h_yes) / n
    p_no  = len(h_no)  / n

    h_after = (p_yes * _uniform_entropy(len(h_yes)) +
               p_no  * _uniform_entropy(len(h_no)))
    return _uniform_entropy(n) - h_after


def _max_belief(belief: Dict[str, float]) -> float:
    return max(belief.values()) if belief else 0.0


def _top_belief_cid(belief: Dict[str, float]) -> Optional[str]:
    return max(belief, key=belief.get) if belief else None


# ─────────────────────────────────────────────────────────────────
# Neo4j Cypher queries
# ─────────────────────────────────────────────────────────────────

# Find conditions food is RISKY_FOR via shortcut edges
CY_FOOD_RISKY = """
MATCH (fp:FoodPhrase)-[r:RISKY_FOR]->(c)
WHERE toLower(fp.name) = toLower($food_phrase)
   OR toLower(fp.name) CONTAINS toLower($food_phrase)
   OR toLower($food_phrase) CONTAINS toLower(fp.name)
WITH c,
     CASE WHEN r.strength = 'high'     THEN 3
          WHEN r.strength = 'moderate' THEN 2
          ELSE 1 END AS priority
WITH c, max(priority) AS priority,
     CASE
       WHEN toLower(c.name) CONTAINS 'diabetes'     THEN 10
       WHEN toLower(c.name) CONTAINS 'hypertension' THEN 9
       WHEN toLower(c.name) CONTAINS 'kidney'       THEN 8
       WHEN toLower(c.name) CONTAINS 'heart'        THEN 8
       WHEN toLower(c.name) CONTAINS 'coronary'     THEN 8
       WHEN toLower(c.name) CONTAINS 'arrhythmia'   THEN 8
       WHEN toLower(c.name) CONTAINS 'atrial'       THEN 8
       WHEN toLower(c.name) CONTAINS 'cholesterol'  THEN 7
       WHEN toLower(c.name) CONTAINS 'allerg'       THEN 7
       WHEN toLower(c.name) CONTAINS 'reflux'       THEN 7
       WHEN toLower(c.name) CONTAINS 'gastritis'    THEN 7
       WHEN toLower(c.name) CONTAINS 'bowel'        THEN 7
       WHEN toLower(c.name) CONTAINS 'ibs'          THEN 7
       WHEN toLower(c.name) CONTAINS 'ulcerative'   THEN 7
       WHEN toLower(c.name) CONTAINS 'crohn'        THEN 7
       WHEN toLower(c.name) CONTAINS 'celiac'       THEN 6
       WHEN toLower(c.name) CONTAINS 'pregnancy'    THEN 6
       WHEN toLower(c.name) CONTAINS 'thyroid'      THEN 5
       ELSE 3 END AS prevalence
RETURN c.id AS cid, c.name AS name, priority, prevalence
ORDER BY priority DESC, prevalence DESC
LIMIT $topk
"""

# Find conditions via ingredient→property→condition chain
CY_FOOD_RISKY_CHAIN = """
MATCH (fp:FoodPhrase)-[:HAS_INGREDIENT]->(i:Ingredient)
      -[:HAS_PROPERTY]->(p:FoodProperty)-[:RISKY_FOR]->(c)
WHERE toLower(fp.name) = toLower($food_phrase)
   OR toLower(fp.name) CONTAINS toLower($food_phrase)
   OR toLower($food_phrase) CONTAINS toLower(fp.name)
RETURN DISTINCT c.id AS cid, c.name AS name, p.name AS via_property
LIMIT $topk
"""

# Find discriminating conditions (those that differ across remaining hypotheses)
# For food task: ask about the condition with highest IG
CY_ALL_CONDITIONS = """
UNWIND $cond_ids AS cid
MATCH (c {id: cid})
RETURN c.id AS cid, c.name AS name
"""

# Find shared IS_A family for remaining conditions (mirrors CY_SHARED_FAMILY)
CY_SHARED_CONDITION_FAMILY = """
UNWIND $cond_ids AS cid
MATCH (c {id: cid})-[:IS_A]->(f:ConditionFamily)
WITH f.id AS family_id, f.name AS family_name,
     collect(cid) AS members, size(collect(cid)) AS coverage
WHERE coverage >= $min_coverage
RETURN family_id, family_name, coverage
ORDER BY coverage DESC, size(family_name) DESC
LIMIT 1
"""

# Check SAFE_FOR: food safe for a condition — via property chain OR direct shortcut
CY_FOOD_SAFE = """
MATCH (fp:FoodPhrase)
WHERE toLower(fp.name) = toLower($food_phrase)
WITH fp
OPTIONAL MATCH (fp)-[:SAFE_FOR]->(c1)
OPTIONAL MATCH (fp)-[:HAS_INGREDIENT]->(:Ingredient)
              -[:HAS_PROPERTY]->(:FoodProperty)-[:SAFE_FOR]->(c2)
WITH [x IN collect(DISTINCT c1) WHERE x IS NOT NULL] +
     [x IN collect(DISTINCT c2) WHERE x IS NOT NULL] AS all_safe
UNWIND all_safe AS c
RETURN DISTINCT c.id AS cid, c.name AS name
LIMIT 30
"""

# Resolve condition by name (for alias lookup)
CY_RESOLVE_CONDITION = """
MATCH (c)
WHERE toLower(c.name) = toLower($name)
   OR toLower(c.name) CONTAINS toLower($name)
   OR toLower($name) CONTAINS toLower(c.name)
RETURN c.id AS cid, c.name AS name
LIMIT 5
"""

# Resolve condition alias
CY_RESOLVE_CONDITION_ALIAS = """
MATCH (a:ConditionAlias)-[:ALIAS_OF]->(c)
WHERE toLower(a.text) = toLower($name)
   OR toLower(a.text) CONTAINS toLower($name)
RETURN c.id AS cid, c.name AS name
LIMIT 3
"""


# ─────────────────────────────────────────────────────────────────
# FoodAmbiguityTask
# ─────────────────────────────────────────────────────────────────

class FoodAmbiguityTask(BaseTask):
    """
    Food safety ambiguity-mitigation agent.

    Implements the POMDP clarification loop from Section 3 of the paper,
    adapted for the food safety domain:

    Given an ambiguous food query ("Can I eat X?"), the agent:
    1. Looks up conditions for which food X is RISKY_FOR in the KG
    2. Iteratively asks the patient about their health conditions
    3. Terminates with OKAY / NOT OKAY / PARTIAL / ABSTAIN

    Mirrors DiagnosticAmbiguityTask exactly — same algorithm,
    different KG direction (food→condition vs symptom→diagnosis).
    """

    name: ClassVar[str] = "food_ambiguity_task"
    chat_name: ClassVar[str] = "FoodAmbiguityTask"
    description: ClassVar[str] = (
        "POMDP food safety agent. Given a food query and patient context, "
        "determines whether the food is OKAY or NOT OKAY for the patient."
    )
    dependencies: ClassVar[List[str]] = []
    inputs: ClassVar[List[str]] = ["food_query", "patient_context"]
    outputs: ClassVar[List[str]] = []
    output_type: ClassVar[bool] = False

    # Neo4j connection params
    uri:      str = "neo4j://127.0.0.1:7687"
    user:     str = "neo4j"
    password: str = "12345678"

    # Private state
    _driver:        Any                  = PrivateAttr(default=None)
    _conditions:    List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _belief:        Dict[str, float]     = PrivateAttr(default_factory=dict)
    _confirmed:     Optional[str]        = PrivateAttr(default=None)
    _denied:        Set[str]             = PrivateAttr(default_factory=set)
    _asked:         List[str]            = PrivateAttr(default_factory=list)
    _question_log:  List[Dict]           = PrivateAttr(default_factory=list)
    _safe_for:      Set[str]             = PrivateAttr(default_factory=set)

    def __init__(self, **data):
        super().__init__(**data)
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    # ── Neo4j helpers ──────────────────────────────────────────

    def _query(self, cypher: str, **params) -> List[Dict]:
        with self._driver.session() as s:
            result = s.run(cypher, **params)
            return [dict(r) for r in result]

    def _reset(self) -> None:
        self._conditions   = []
        self._belief       = {}
        self._confirmed    = None
        self._denied       = set()
        self._asked        = []
        self._question_log = []
        self._safe_for     = set()

    # ── KG lookup ─────────────────────────────────────────────

    def _find_risky_conditions(self, food_phrase: str) -> List[Dict]:
        """
        Retrieve conditions for which food is RISKY_FOR.
        Uses shortcut edges first, falls back to chain query.
        Filters out conditions patient has already denied.
        """
        rows = self._query(
            CY_FOOD_RISKY,
            food_phrase=food_phrase,
            topk=TOPK_CONDITIONS,
        )

        if not rows:
            # Fallback: ingredient→property→condition chain
            rows = self._query(
                CY_FOOD_RISKY_CHAIN,
                food_phrase=food_phrase,
                topk=TOPK_CONDITIONS,
            )

        # Deduplicate by cid
        seen = set()
        conditions = []
        for r in rows:
            if r["cid"] and r["cid"] not in seen:
                seen.add(r["cid"])
                conditions.append({
                    "cid":  r["cid"],
                    "name": r.get("name") or r["cid"],
                })

        return conditions

    def _find_safe_conditions(self, food_phrase: str) -> Set[str]:
        """Find conditions food is explicitly SAFE_FOR.
        Returns lowercased condition NAMES (not IDs) to avoid CID mismatch."""
        rows = self._query(CY_FOOD_SAFE, food_phrase=food_phrase)
        return {r["name"].strip().lower() for r in rows if r.get("name")}

    def _resolve_condition(self, condition_text: str) -> Optional[str]:
        """
        Resolve a condition text string to a condition node ID.
        Tries alias lookup first, then direct name match.
        """
        if not condition_text:
            return None

        norm_text = _norm_condition(condition_text)

        # Try alias lookup
        rows = self._query(
            CY_RESOLVE_CONDITION_ALIAS,
            name=norm_text,
        )
        if rows:
            return rows[0]["cid"]

        # Try direct name match
        rows = self._query(
            CY_RESOLVE_CONDITION,
            name=norm_text,
        )
        if rows:
            return rows[0]["cid"]

        return None

    def _active_conditions(self) -> List[Dict]:
        """Return conditions not yet denied and not yet confirmed.
        Limited to TOPK_ASK for POMDP resolution within MAX_TURNS."""
        if self._confirmed:
            return [c for c in self._conditions if c["cid"] == self._confirmed]
        denied_cids = self._denied
        active = [c for c in self._conditions if c["cid"] not in denied_cids]
        return active[:TOPK_ASK]  # keep top-TOPK_ASK by strength order

    # ── POMDP logic ────────────────────────────────────────────

    def _select_best_question(self) -> Tuple[Optional[str], float]:
        """
        Select condition with maximum information gain.
        Mirrors _select_questions() from DiagnosticAmbiguityTask.
        Returns (cid, ig) of best question.
        """
        active = self._active_conditions()
        if not active:
            return None, 0.0

        best_cid, best_ig = None, -1.0
        for c in active:
            if c["cid"] in self._asked:
                continue
            ig = _ig_condition(active, c["cid"])
            if ig > best_ig:
                best_ig  = ig
                best_cid = c["cid"]

        return best_cid, best_ig

    def _should_stop(self, delta: float) -> Tuple[bool, str]:
        """
        Parameter-free stopping conditions (Algorithm 1):

        Condition 1: max(b) > 1 - delta
            Belief concentrated → ANSWER
        Condition 2: IG(v*) = 0
            No further question reduces entropy → PARTIAL or ABSTAIN
        """
        # Fast exit: confirmed risky condition
        if self._confirmed is not None:
            return True, "confirmed_risky"

        # Fast exit: all conditions denied → food is OKAY
        active = self._active_conditions()
        if not active:
            return True, "all_denied"

        # Condition 1: belief concentration
        max_b = _max_belief(self._belief)
        if max_b > 1.0 - delta:
            return True, "belief_concentrated"

        # Condition 2: zero IG
        _, best_ig = self._select_best_question()
        if best_ig <= 0.0:
            return True, "zero_ig"

        return False, ""

    def _assign_agent_state(
        self, stop_reason: str, delta: float
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Assign ANSWER/PARTIAL/ABSTAIN and find IS_A family.
        Mirrors _assign_agent_state() from DiagnosticAmbiguityTask.
        """
        # Confirmed risky condition → NOT OKAY
        if stop_reason == "confirmed_risky":
            return "ANSWER", None, None

        # All conditions denied → food is OKAY
        if stop_reason == "all_denied":
            return "ANSWER", None, None

        active = self._active_conditions()

        # Belief concentrated → ANSWER with top condition
        if stop_reason == "belief_concentrated":
            return "ANSWER", None, None

        # Zero IG or max turns → check IS_A family
        if stop_reason in ("zero_ig", "max_turns") and len(active) > 0:
            cond_ids    = [c["cid"] for c in active]
            min_cov     = max(1, math.ceil(len(cond_ids) / 2))

            rows = self._query(
                CY_SHARED_CONDITION_FAMILY,
                cond_ids=cond_ids,
                min_coverage=min_cov,
            )
            if rows:
                return "PARTIAL", rows[0]["family_id"], rows[0]["family_name"]
            return "ABSTAIN", None, None

        return "ABSTAIN", None, None

    def _condition_name(self, cid: str) -> str:
        for c in self._conditions:
            if c["cid"] == cid:
                return c["name"]
        return cid

    def _rephrase_question(
        self, condition_name: str, client: Any
    ) -> str:
        """
        Use LLM to rephrase KG condition name into patient-friendly language.
        Mirrors _rephrase_question() from DiagnosticAmbiguityTask.
        """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                max_tokens=60,
                messages=[{
                    "role": "system",
                    "content": (
                        "Rephrase the medical condition name into a simple "
                        "yes/no patient question. Return ONLY the question, "
                        "no explanation."
                    ),
                }, {
                    "role": "user",
                    "content": f"Condition: {condition_name}",
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"Do you have {condition_name}?"

    def _generate_final_response(
        self,
        food_phrase: str,
        decision: str,
        confirmed_condition: Optional[str],
        active_conditions: List[Dict],
        family_name: Optional[str],
        question_log: List[Dict],
        client: Any,
        model: str,
    ) -> str:
        """
        Generate natural language response via LLM.
        LLM is confined to articulation — cannot change the decision.
        Mirrors final response generation in DiagnosticAmbiguityTask.
        """
        if client is None:
            return decision

        qa_text = ""
        for q in question_log:
            ans = "Yes" if q["answer"] else "No"
            qa_text += f"  Q: {q['question']}  A: {ans}\n"

        if confirmed_condition:
            context = (
                f"The patient asked about: {food_phrase}\n"
                f"Patient confirmed they have: {confirmed_condition}\n"
                f"Decision: NOT OKAY — this food is not safe for the patient.\n"
                f"Clarification dialogue:\n{qa_text}"
            )
        elif not active_conditions:
            context = (
                f"The patient asked about: {food_phrase}\n"
                f"No dietary restrictions identified that conflict with this food.\n"
                f"Decision: OKAY — this food appears safe for the patient.\n"
                f"Clarification dialogue:\n{qa_text}"
            )
        elif family_name:
            names = [c["name"] for c in active_conditions[:3]]
            context = (
                f"The patient asked about: {food_phrase}\n"
                f"Potential concern area: {family_name} "
                f"(conditions: {', '.join(names)})\n"
                f"Decision: PARTIAL — further clinical evaluation recommended.\n"
                f"Clarification dialogue:\n{qa_text}"
            )
        else:
            context = (
                f"The patient asked about: {food_phrase}\n"
                f"Decision: ABSTAIN — insufficient information to determine safety.\n"
                f"Clarification dialogue:\n{qa_text}"
            )

        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=200,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a dietary safety assistant. "
                        "Given the decision and context, write a clear, "
                        "empathetic 2-3 sentence response to the patient. "
                        "Do NOT change the decision. Do NOT give medical advice."
                    ),
                }, {
                    "role": "user",
                    "content": context,
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return decision

    # ── openCHA interface ─────────────────────────────────────

    def _execute(self, inputs: List[Any]) -> str:
        """Standard openCHA entry point."""
        t0 = time.time()
        try:
            food_query      = str(inputs[0]) if inputs else ""
            patient_context = str(inputs[1]) if len(inputs) > 1 else ""

            result = self.run_benchmark_case(
                food_query=food_query,
                patient_context=patient_context,
                ground_truth=patient_context,
            )

            return json.dumps({
                "ok":          True,
                "decision":    result["decision"],
                "agent_state": result["agent_state"],
                "response":    result["response"],
                "diagnostics": {
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    "turns":      result["turns"],
                },
            })
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    # ── POMDP benchmark runner (main method) ───────────────────

    def run_benchmark_case(
        self,
        food_query:      str,
        patient_context: str,
        ground_truth:    str,
        delta:           float = DELTA,
        openai_client:   Any   = None,
        model:           str   = "gpt-4o",
        rephrase:        bool  = False,
        epsilon:         float = DEFAULT_EPSILON,   # kept for compat
        t_max:           int   = DEFAULT_T_MAX,     # kept for compat
    ) -> Dict[str, Any]:
        """
        POMDP ambiguity-mitigation loop for food safety.

        Stopping governed by two parameter-free conditions:
        1. Belief concentration: max(b) > 1 - delta
        2. Zero information gain: IG(v*) = 0

        Agent state on exit:
            ANSWER  — decision is clear (OK or NOT OK)
            PARTIAL — family-level resolution, pass to LLM
            ABSTAIN — cannot resolve safely, skip LLM
        """
        self._reset()

        # ── Extract food phrase from query ──────────────────
        food_phrase = _extract_food_phrase(food_query)

        # Detect garbage extraction — phrase too short or meaningless
        GARBAGE_PHRASES = {"it", "it fine", "it safe", "it okay", "it healthy",
                           "would it", "is it", "that", "this", "fine", "okay",
                           "safe", "healthy", "good", "bad", "a problem", "alright"}
        if len(food_phrase) <= 3 or food_phrase.lower() in GARBAGE_PHRASES:
            # Cannot extract food — return ABSTAIN immediately
            return {
                "food_query":        food_query,
                "food_phrase":       food_phrase,
                "ground_truth":      ground_truth,
                "decision":          "ABSTAIN",
                "response":          "ABSTAIN",
                "clarified_query":   food_query,
                "agent_state":       "ABSTAIN",
                "family_id":         None,
                "family_name":       None,
                "max_belief":        0.0,
                "n_hypotheses_init":  0,
                "n_hypotheses_final": 0,
                "entropy_init":       0.0,
                "entropy_final":      0.0,
                "candidates_init":    [],
                "candidates_final":   [],
                "turns":              0,
                "stop_reason":        "garbage_phrase",
                "abstained":          True,
                "question_log":       [],
                "confirmed_condition": None,
                "ground_truth_cid":   None,
            }
        ground_truth_cid = None

        # ── Build hypothesis set H from KG ──────────────────
        # H = {conditions food is RISKY_FOR}
        self._conditions = self._find_risky_conditions(food_phrase)
        self._safe_for   = self._find_safe_conditions(food_phrase)

        n_init       = len(self._conditions)
        entropy_init = _uniform_entropy(n_init)
        conds_init   = [{"name": c["name"]} for c in self._conditions]

        # Initialize uniform belief
        self._belief = _belief_init(self._conditions)

        # ── Resolve ground-truth condition (oracle) ──────────
        gt_norm = _norm_condition(ground_truth)
        if gt_norm and gt_norm != "no restrictions":
            ground_truth_cid = self._resolve_condition(gt_norm)

        # ── POMDP clarification loop ─────────────────────────
        turns       = 0
        stop_reason = "single"

        while True:
            should_stop, reason = self._should_stop(delta)
            if should_stop:
                stop_reason = reason
                break

            # Select max-IG condition to ask about
            best_cid, best_ig = self._select_best_question()
            if best_cid is None or best_ig <= 0.0:
                stop_reason = "zero_ig"
                break

            condition_name = self._condition_name(best_cid)

            # Build question text
            if rephrase and openai_client is not None:
                question_text = self._rephrase_question(
                    condition_name, openai_client
                )
            else:
                question_text = f"Do you have {condition_name}?"

            # Oracle answer: does patient have this condition?
            answer = _oracle_answer(
                asked_cid=best_cid,
                asked_name=condition_name,
                ground_truth_cid=ground_truth_cid,
                ground_truth_text=gt_norm,
                all_conditions=self._conditions,
            )

            # Update belief via Bayes rule
            active_before = self._active_conditions()
            self._belief = _belief_update(
                self._belief, active_before, best_cid, answer
            )

            # Update hard sets
            if answer:
                self._confirmed = best_cid
            else:
                self._denied.add(best_cid)

            self._question_log.append({
                "cid":      best_cid,
                "name":     condition_name,
                "question": question_text,
                "answer":   answer,
                "ig":       round(best_ig, 4),
                "max_b":    round(_max_belief(self._belief), 4),
            })
            self._asked.append(best_cid)

            # Renormalize belief after removal
            surviving = {c["cid"] for c in self._active_conditions()}
            self._belief = {
                k: v for k, v in self._belief.items()
                if k in surviving
            }
            total_b = sum(self._belief.values())
            if total_b > 0:
                self._belief = {k: v / total_b for k, v in self._belief.items()}
            elif self._active_conditions():
                self._belief = _belief_init(self._active_conditions())

            turns += 1

            # Check confirmed_risky BEFORE max_turns
            # (patient may confirm on the final turn)
            if self._confirmed is not None:
                stop_reason = "confirmed_risky"
                break

            # Cap at MAX_TURNS
            if turns >= MAX_TURNS:
                stop_reason = "max_turns"
                break

        # ── Assign agent state ───────────────────────────────
        agent_state, family_id, family_name = self._assign_agent_state(
            stop_reason, delta
        )

        # ── Determine food safety decision ───────────────────
        active_final = self._active_conditions()
        if stop_reason == "confirmed_risky" and self._confirmed:
            # Compare by condition NAME (not ID) to avoid CID mismatch
            confirmed_name = self._condition_name(self._confirmed).strip().lower()
            if confirmed_name in self._safe_for:
                decision            = "OKAY"
                confirmed_cond_name = self._condition_name(self._confirmed)
            else:
                decision            = "NOT OKAY"
                confirmed_cond_name = self._condition_name(self._confirmed)
        elif stop_reason == "all_denied" or (
            agent_state == "ANSWER" and not self._confirmed
        ):
            decision            = "OKAY"
            confirmed_cond_name = None
        elif agent_state == "PARTIAL":
            decision            = "PARTIAL"
            confirmed_cond_name = None
        else:
            decision            = "ABSTAIN"
            confirmed_cond_name = None

        # ── Final metrics ────────────────────────────────────
        entropy_final  = _uniform_entropy(len(active_final))
        max_belief_val = round(_max_belief(self._belief), 4)
        conds_final    = [{"name": c["name"]} for c in active_final[:10]]

        # ── Build clarified query for LLM ────────────────────
        qa_text = ""
        for q in self._question_log:
            ans = "Yes" if q["answer"] else "No"
            qa_text += f"  - {q['name']}: {ans}\n"

        if qa_text:
            clarified_query = (
                f"Food query: {food_query}\n"
                f"Clarification answers:\n{qa_text}"
            )
        else:
            clarified_query = f"Food query: {food_query}"

        if active_final and agent_state in ("PARTIAL", "ANSWER"):
            names = [c["name"] for c in active_final[:3]]
            clarified_query += f"\nKG risk conditions: {', '.join(names)}."
        if family_name:
            clarified_query += f"\nCondition family: {family_name}."

        # ── Generate LLM response ────────────────────────────
        response = "ABSTAIN"
        if agent_state != "ABSTAIN" and openai_client is not None:
            response = self._generate_final_response(
                food_phrase=food_phrase,
                decision=decision,
                confirmed_condition=confirmed_cond_name,
                active_conditions=active_final,
                family_name=family_name,
                question_log=self._question_log,
                client=openai_client,
                model=model,
            )
        elif agent_state != "ABSTAIN":
            response = decision

        return {
            # ── Core outputs ──
            "food_query":     food_query,
            "food_phrase":    food_phrase,
            "ground_truth":   ground_truth,
            "decision":       decision,
            "response":       response,
            "clarified_query": clarified_query,

            # ── Agent state ──
            "agent_state":    agent_state,
            "family_id":      family_id,
            "family_name":    family_name,
            "max_belief":     max_belief_val,

            # ── Hypothesis metrics ──
            "n_hypotheses_init":  n_init,
            "n_hypotheses_final": len(active_final),
            "entropy_init":       round(entropy_init,  4),
            "entropy_final":      round(entropy_final, 4),
            "candidates_init":    conds_init,
            "candidates_final":   conds_final,

            # ── Clarification metrics ──
            "turns":         turns,
            "stop_reason":   stop_reason,
            "abstained":     agent_state == "ABSTAIN",
            "question_log":  list(self._question_log),

            # ── Ground truth tracking ──
            "confirmed_condition": confirmed_cond_name,
            "ground_truth_cid":    ground_truth_cid,
        }


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def _extract_food_phrase(query: str) -> str:
    """
    Extract core food phrase from natural language query.
    Comprehensive extraction handling embedded conditions and complex queries.
    """
    q = re.sub(r"\s+", " ", query.strip().lower())

    # Remove question prefixes — including queries with embedded conditions
    for pat in [
        r"^can i eat\s+", r"^can i drink\s+", r"^can i have\s+",
        r"^can i enjoy\s+", r"^can i include\s+", r"^can i try\s+",
        r"^can i snack on\s+",
        # Embedded condition: "can a person with X have Y" → extract Y
        r"^can a person with[\w\s-]+have\s+",
        r"^can a person with[\w\s-]+eat\s+",
        r"^can someone with[\w\s-]+enjoy\s+",
        r"^can someone with[\w\s-]+eat\s+",
        r"^can someone with[\w\s-]+have\s+",
        r"^should someone with[\w\s-]+avoid\s+",
        r"^should someone with[\w\s-]+eat\s+",
        r"^could i have\s+",
        r"^would it be okay to\s+\w+\s+a?\s*",
        r"^would it be okay if i\s+\w+\s+a?\s*",
        r"^would it be safe to\s+\w+\s+a?\s*",
        r"^would it be alright to\s+\w+\s+a?\s*",
        r"^would it be advisable to\s+\w+\s+a?\s*",
        r"^would it be okay to have\s+a?\s*",
        r"^would it be fine to\s+\w+\s+a?\s*",
        r"^would it be good to\s+\w+\s+a?\s*",
        r"^would it be harmful to\s+\w+\s+",
        r"^is it okay to eat\s+", r"^is it okay to drink\s+",
        r"^is it okay if i\s+\w+\s+", r"^is it safe to\s+\w+\s+",
        r"^is it safe for me to\s+\w+\s+", r"^is it advisable to\s+\w+\s+",
        r"^is it alright to\s+\w+\s+", r"^is it wise to\s+\w+\s+",
        r"^it okay to\s+\w+\s+a?\s*", r"^it wise to\s+\w+\s+",
        r"^it safe to\s+\w+\s+",
        r"^should i avoid\s+", r"^should i steer clear of\s+",
        r"^should i stay away from\s+",
        r"^i m craving\s+", r"^i m craving\s+a?\s*",
        r"^i\'m craving\s+a?\s*", r"^i\'m craving\s+some\s*",
        r"^i want to\s+\w+\s+", r"^would a\s+", r"^would an\s+",
        r"^would having\s+a?\s*", r"^would eating\s+",
        r"^is a\s+", r"^is an\s+", r"^are\s+", r"^is\s+",
        r"^a bowl of\s+", r"^a cup of\s+", r"^a glass of\s+",
        r"^a slice of\s+", r"^a bag of\s+", r"^a piece of\s+",
        r"^a small\s+", r"^a few\s+",
        r"^some\s+", r"^a\s+", r"^an\s+", r"^on\s+",
        r"^i m in the mood for\s+a?\s*", r"^i\'m in the mood for\s+a?\s*",
        r"^in the mood for\s+a?\s*",
        r"^i m thinking about\s+", r"^i\'m thinking about\s+",
        r"^i m thinking about trying\s+", r"^i\'m thinking about trying\s+",
        r"^i m thinking of\s+\w+\s+", r"^i\'m thinking of\s+\w+\s+",
        r"^i m thinking of having\s+a?\s*", r"^i\'m thinking of having\s+a?\s*",
        r"^i m thinking of trying\s+", r"^i\'m thinking of trying\s+",
        r"^i m planning to eat\s+",
        r"^i d like to have\s+", r"^i\'d like to have\s+",
        r"^i want to eat\s+", r"^i want to try\s+",
        r"^i m going to have\s+", r"^i\'m going to have\s+",
        r"^i m considering\s+a?\s*", r"^i\'m considering\s+a?\s*",
        r"^i m looking forward to\s+a?\s*", r"^i\'m looking forward to\s+a?\s*",
        r"^i love\s+\w+,?\s+but\s+is\s+it\s+",
        r"^adding\s+", r"^drinking\s+", r"^eating\s+", r"^having\s+",
        r"^is drinking\s+", r"^is eating\s+", r"^is adding\s+",
        r"^is consuming\s+",
    ]:
        q = re.sub(pat, "", q)

    # Remove trailing context and punctuation
    for pat in [
        # Condition-specific trailing patterns
        r"\s+with\s+my\s+[\w\s]+$",
        r"\s+with\s+(my\s+)?(diabetes|ulcerative colitis|lactose intolerance|celiac disease|ibs|irritable bowel|gerd|hypertension|high blood pressure|high cholesterol|hypercholesterolemia|cholesterol|crohn|heart disease|kidney disease|type 2 diabetes|type 1 diabetes|prediabetes|hypothyroidism|hyperthyroidism|cardiovascular|acid reflux|gastritis|congestive heart|metabolic syndrome|pre-diabetes|pregnancy|gestational diabetes|gluten intolerance|arrhythmia|ulcerative)[\w\s]*$",
        r"\s+for\s+(diabetes|celiac|gerd|ibs|hypertension|heart|cholesterol|kidney|lactose|pregnancy|prediabetes|acid reflux|gastritis|type 2|type 1|managing|pre-diabetes|metabolic|irritable|ulcerative|crohn|arrhythmia|gluten)[\w\s]*$",
        r"\s+when\s+(managing|dealing with|living with|i have|i\'m having|i\'m managing)[\w\s]*$",
        r"\s+if\s+i\'m\s+(managing|dealing with|living with|having|dealing)[\w\s]*$",
        r"\s+if\s+i\s+have.*$",
        r"\s+a\s+good\s+(breakfast|lunch|dinner|snack|meal|choice|option|idea)[\w\s]*$",
        r"\s+a\s+suitable\s*(option|choice|meal|snack)[\w\s]*$",
        r"\s+suitable\s*(option|choice|for)[\w\s]*$",
        r"\s+recommended\s*(for|to|by|\?)[\w\s]*$",
        r"\s+advisable\s*(for|\?)?[\w\s]*$",
        r"\s+help\s+(my|the|me|with)[\w\s]*$",
        r"\s+help\s+my[\w\s]*$",
        r"\s+a risk for me[\w\s]*$",
        r"\s+each\s+(morning|evening|day)[\w\s]*$",
        r"\s+at\s+the\s+(movies|theater|cinema)[\w\s]*$",
        r"\s+on\s+the\s+go[\w\s]*$",
        r"\s+each morning[\w\s]*$",
        # General trailing patterns
        r"\s+for someone with.*$", r"\s+for my condition.*$",
        r"\s+is that okay.*$", r"\s+be a problem.*$", r"\s+a problem.*$",
        r"\s+be okay.*$", r"\s+be safe.*$", r"\s+today.*$",
        r"\s+tonight.*$", r"\s+now.*$", r"\s+for breakfast.*$",
        r"\s+for lunch.*$", r"\s+for dinner.*$",
        r"\s+be alright.*$", r"\s+be suitable.*$",
        r"\s+when i m.*$", r"\s+with.*condition.*$",
        r"\s+before my.*$", r"\s+after my.*$",
        r"\s+in the morning.*$", r"\s+in the evening.*$",
        r"\s+okay\??$", r"\s+fine\??$", r"\s+alright\??$",
        r"\s+is that.*$", r"\s+would it.*$",
        r"\s+a good choice.*$", r"\s+a good option.*$",
        r"\s+a bad idea.*$", r"\s+be harmful.*$",
        r"\s+be beneficial.*$", r"\s+be bad.*$",
        r"\s+appropriate.*$", r"\s+suitable for.*$",
        r"\s+with my.*$", r"\s+for my.*$",
        r"\s+to indulge.*$", r"\s+to snack on.*$",
        r"\s+should i skip.*$", r"\s+skip it.*$",
        r"\s+is it okay.*$", r"\s+is that fine.*$",
        r"\s+safe for.*$", r"\s+good for.*$", r"\s+safe\??$",
        r"\?+$", r"\.+$",
    ]:
        q = re.sub(pat, "", q)

    return re.sub(r"\s+", " ", q).strip()


def _oracle_answer(
    asked_cid:        str,
    asked_name:       str,
    ground_truth_cid: Optional[str],
    ground_truth_text: str,
    all_conditions:   List[Dict],
) -> bool:
    """
    Oracle answer for benchmark evaluation.
    Returns True if the patient has the asked condition.

    Mirrors oracle logic in DiagnosticAmbiguityTask:
    answer = asked_cui in full_symptom_cuis
    """
    if not ground_truth_cid or not ground_truth_text:
        return False

    # Exact ID match
    if asked_cid == ground_truth_cid:
        return True

    # Fuzzy name match (handles alias variants)
    gt_norm   = _norm_condition(ground_truth_text)
    ask_norm  = _norm_condition(asked_name)

    if gt_norm and ask_norm:
        if gt_norm in ask_norm or ask_norm in gt_norm:
            return True

    return False


# ─────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = FoodAmbiguityTask(password="12345678")

    print("Testing food POMDP: 'Can I eat white rice?' | Patient: Type 2 Diabetes")
    result = task.run_benchmark_case(
        food_query="Can I eat white rice?",
        patient_context="type 2 diabetes",
        ground_truth="type 2 diabetes",
        delta=0.05,
    )

    print(f"Decision      : {result['decision']}")
    print(f"Agent state   : {result['agent_state']}")
    print(f"Turns         : {result['turns']}")
    print(f"Stop reason   : {result['stop_reason']}")
    print(f"H_init        : {result['n_hypotheses_init']}")
    print(f"H_final       : {result['n_hypotheses_final']}")
    print(f"Question log  : {result['question_log']}")