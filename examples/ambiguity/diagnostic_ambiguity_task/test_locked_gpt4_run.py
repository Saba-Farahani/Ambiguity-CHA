"""
test_locked_gpt4_run.py
-----------------------
POMDP ambiguity-mitigation agent evaluation on the test set.

Agent states:
    ANSWER  — belief concentrated (max_b > 0.95), pass to LLM
    PARTIAL — IG=0, family consensus found, pass to LLM with family hint
    ABSTAIN — IG=0, no family consensus, skip LLM

Stopping is governed by two parameter-free conditions (no t_max, no epsilon):
    1. max(b) > 1 - delta   (Howard 1966, delta=0.05)
    2. IG = 0               (VOI theory)
"""

import ast
import json
import os
import re
import math
from pathlib import Path

import pandas as pd
from openai import OpenAI
from openCHA.tasks.ambiguity.diagnostic_ambiguity_task import DiagnosticAmbiguityTask


BASE_DIR = Path(__file__).resolve().parents[3]

TEST_CSV = BASE_DIR / "data" / "test_patient_dx_symptoms.csv"
OUT_CSV  = BASE_DIR / "results" / "diagnostic_ambiguity_task" / \
           "test_pomdp_outputs.csv"

MODEL     = "gpt-4o"
DELTA     = 0.05        # Clinical significance threshold (Howard 1966)
MAX_ROWS  = None
REPHRASE_QUESTIONS = False


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def uniform_entropy(n: int) -> float:
    return math.log2(n) if n > 1 else 0.0


def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        val = ast.literal_eval(str(x))
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
    except Exception:
        pass
    return []


def candidate_names(items):
    out = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
        elif isinstance(item, str):
            out.append(str(item))
    return out


def strip_code_fences(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def normalize_diag_list(preds):
    cleaned = []
    seen    = set()
    for p in preds:
        p   = str(p).strip()
        key = p.lower()
        if p and key not in seen:
            cleaned.append(p)
            seen.add(key)
    return cleaned[:5]


# ─────────────────────────────────────────────────────────
# LLM prompts — one per agent state
# ─────────────────────────────────────────────────────────

def build_prompt_answer(
    conversation:    str,
    clarified_query: str,
    candidates_final,
    candidates_init,
) -> str:
    """
    ANSWER state: belief is concentrated.
    LLM receives clarified query + KG candidates.
    """
    final_names = candidate_names(candidates_final)
    init_names  = candidate_names(candidates_init)
    cand_block  = final_names if final_names else init_names
    cand_text   = ", ".join(cand_block[:10]) if cand_block else "None available"

    return f"""
You are a careful physician-style diagnostic reasoning assistant.

The ambiguity-mitigation agent has clarified the patient query and identified
high-confidence candidate diagnoses from the knowledge graph.

Your task:
1. Return the single most likely diagnosis.
2. Return a ranked top-5 list.

Guidelines:
- The agent has already narrowed the hypothesis space — prefer diagnoses
  from the KG candidates.
- Be clinically careful and conservative.
- Return ONLY valid JSON with this exact schema:
{{
  "prediction_top1": "single diagnosis name",
  "prediction_top5": ["diag1", "diag2", "diag3", "diag4", "diag5"]
}}

Original conversation:
{conversation}

Clarified case summary (with clarification answers):
{clarified_query}

Knowledge-graph candidate diagnoses (high confidence):
{cand_text}

Return the JSON only.
""".strip()


def build_prompt_partial(
    conversation:    str,
    clarified_query: str,
    candidates_final,
    candidates_init,
    family_name:     str,
) -> str:
    """
    PARTIAL state: IG=0, family consensus found.
    LLM receives clarified query + family constraint + KG candidates.
    """
    final_names = candidate_names(candidates_final)
    init_names  = candidate_names(candidates_init)
    cand_block  = final_names if final_names else init_names
    cand_text   = ", ".join(cand_block[:10]) if cand_block else "None available"

    return f"""
You are a careful physician-style diagnostic reasoning assistant.

The ambiguity-mitigation agent has narrowed the diagnosis to the clinical
family "{family_name}" but cannot distinguish between the remaining candidates
without additional information. The candidates below are all within this family.

Your task:
1. Return the single most likely diagnosis within this clinical family.
2. Return a ranked top-5 list, preferring candidates from within the family.

Guidelines:
- Prefer diagnoses from the KG candidates listed below.
- The clinical family context should guide your ranking.
- Be clinically careful and conservative.
- Return ONLY valid JSON with this exact schema:
{{
  "prediction_top1": "single diagnosis name",
  "prediction_top5": ["diag1", "diag2", "diag3", "diag4", "diag5"]
}}

Original conversation:
{conversation}

Clarified case summary:
{clarified_query}

Clinical family: {family_name}

Knowledge-graph candidate diagnoses within this family:
{cand_text}

Return the JSON only.
""".strip()


def build_prompt_abstain_fallback(
    conversation:    str,
    clarified_query: str,
    candidates_init,
) -> str:
    """
    ABSTAIN state fallback: agent could not resolve ambiguity.
    LLM receives initial candidates only — no clarification signal.
    Used only to measure what GPT would have said (safety analysis).
    Primary output for ABSTAIN cases is "ABSTAIN".
    """
    init_names = candidate_names(candidates_init)
    cand_text  = ", ".join(init_names[:10]) if init_names else "None available"

    return f"""
You are a careful physician-style diagnostic reasoning assistant.

NOTE: The ambiguity-mitigation agent could not resolve the diagnostic
ambiguity in this case — insufficient clinical information was available.
This response is a fallback for research analysis only.

Your task:
1. Return the single most likely diagnosis given the limited information.
2. Return a ranked top-5 list.

Original conversation:
{conversation}

Initial KG candidates (pre-clarification):
{cand_text}

Return ONLY valid JSON:
{{
  "prediction_top1": "single diagnosis name",
  "prediction_top5": ["diag1", "diag2", "diag3", "diag4", "diag5"]
}}
""".strip()


# ─────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────

def call_llm(client: OpenAI, prompt: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250,
    )
    raw         = resp.choices[0].message.content.strip()
    parsed_text = strip_code_fences(raw)

    prediction_top1 = ""
    prediction_top5 = []

    try:
        obj             = json.loads(parsed_text)
        prediction_top1 = str(obj.get("prediction_top1", "")).strip()
        prediction_top5 = normalize_diag_list(obj.get("prediction_top5", []))
    except Exception:
        prediction_top1 = parsed_text
        prediction_top5 = []

    return raw, prediction_top1, prediction_top5


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    neo4j_pass = os.getenv("NEO4J_PASSWORD", "")  # set NEO4J_PASSWORD env var

    client = OpenAI(api_key=api_key)
    task   = DiagnosticAmbiguityTask(password=neo4j_pass)

    df = pd.read_csv(TEST_CSV)
    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    print(f"Loaded test rows  : {len(df)}")
    print(f"Model             : {MODEL}")
    print(f"Delta (VOI)       : {DELTA}  → acts when P(dx) > {1-DELTA:.0%}")
    print(f"Stopping          : max(b) > {1-DELTA} OR IG = 0  [no t_max, no epsilon]")

    rows_out = []

    # State counters for progress reporting
    state_counts = {"ANSWER": 0, "PARTIAL": 0, "ABSTAIN": 0}

    for i, row in df.iterrows():
        visible_col = (
            "REMAINING_SYMPTOMS"
            if "REMAINING_SYMPTOMS" in row.index
            else "CURRENT_SYMPTOMS"
        )

        remaining_symptoms = parse_list(row[visible_col])
        full_symptoms      = parse_list(row["FULL_SYMPTOMS"])
        conversation       = str(row["Conversation"])
        ground_truth       = str(row["PATHOLOGY"])

        # ── Run POMDP agent ───────────────────────────────
        result = task.run_benchmark_case(
            remaining_symptoms=remaining_symptoms,
            full_symptoms=full_symptoms,
            conversation=conversation,
            ground_truth=ground_truth,
            delta=DELTA,
            rephrase=REPHRASE_QUESTIONS,
        )

        agent_state      = result["agent_state"]
        clarified_query  = result["clarified_query"]
        candidates_init  = result["candidates_init"]
        candidates_final = result["candidates_final"]
        family_name      = result.get("family_name") or ""

        state_counts[agent_state] = state_counts.get(agent_state, 0) + 1

        # ── Call LLM based on agent state ─────────────────
        raw_llm         = ""
        prediction_top1 = ""
        prediction_top5 = []
        llm_called      = False

        if agent_state == "ANSWER":
            prompt = build_prompt_answer(
                conversation, clarified_query,
                candidates_final, candidates_init,
            )
            raw_llm, prediction_top1, prediction_top5 = call_llm(
                client, prompt
            )
            llm_called = True

        elif agent_state == "PARTIAL":
            prompt = build_prompt_partial(
                conversation, clarified_query,
                candidates_final, candidates_init,
                family_name,
            )
            raw_llm, prediction_top1, prediction_top5 = call_llm(
                client, prompt
            )
            llm_called = True

        elif agent_state == "ABSTAIN":
            # Primary output is ABSTAIN — record what GPT would have said
            # for safety analysis but do not use it as main prediction
            prompt = build_prompt_abstain_fallback(
                conversation, clarified_query, candidates_init,
            )
            raw_llm, prediction_top1, prediction_top5 = call_llm(
                client, prompt
            )
            llm_called       = True
            # Mark as abstained — main prediction is ABSTAIN
            prediction_top1  = "ABSTAIN"
            prediction_top5  = []

        # ── Save row ──────────────────────────────────────
        rows_out.append({
            "row_id":            i,
            "ground_truth":      ground_truth,
            "conversation":      conversation,
            "full_symptoms":     json.dumps(full_symptoms),
            "remaining_symptoms":json.dumps(remaining_symptoms),
            "resolved_initial_symptoms": json.dumps(
                result.get("resolved_initial_symptoms", [])
            ),

            # Agent state
            "agent_state":       agent_state,
            "family_name":       family_name,
            "max_belief":        result.get("max_belief", 0.0),

            # Hypothesis metrics
            "n_hypotheses_init": result["n_hypotheses_init"],
            "entropy_init":      result["entropy_init"],
            "candidates_init":   json.dumps(candidates_init),
            "n_hypotheses_final":result["n_hypotheses_final"],
            "entropy_final":     result["entropy_final"],
            "candidates_final":  json.dumps(candidates_final),

            # Clarification metrics
            "turns":             result["turns"],
            "stop_reason":       result["stop_reason"],
            "abstained":         result["abstained"],
            "question_log":      json.dumps(result["question_log"]),

            # LLM output
            "clarified_query":   clarified_query,
            "top_candidate":     result.get("top_candidate", ""),
            "llm_called":        llm_called,
            "final_prompt":      prompt if llm_called else "",
            "raw_llm_output":    raw_llm,
            "prediction_top1":   prediction_top1,
            "prediction_top5":   json.dumps(prediction_top5),
        })

        # ── Progress ──────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"ROW {i+1}/{len(df)}")
        print(f"{'='*70}")
        print(f"GT              : {ground_truth}")
        print(f"Agent state     : {agent_state}")
        print(f"Family          : {family_name or 'N/A'}")
        print(f"Max belief      : {result.get('max_belief', 0.0):.4f}")
        print(f"Turns           : {result['turns']}")
        print(f"Stop reason     : {result['stop_reason']}")
        print(f"Candidates final: {candidate_names(candidates_final)}")
        print(f"Prediction top1 : {prediction_top1}")
        print(f"State counts    : {state_counts}")

    # ── Save ──────────────────────────────────────────────
    out_df = pd.DataFrame(rows_out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # ── Final summary ─────────────────────────────────────
    total = len(out_df)
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total cases      : {total}")
    print(f"ANSWER           : {state_counts.get('ANSWER', 0)} "
          f"({100*state_counts.get('ANSWER',0)/total:.1f}%)")
    print(f"PARTIAL          : {state_counts.get('PARTIAL', 0)} "
          f"({100*state_counts.get('PARTIAL',0)/total:.1f}%)")
    print(f"ABSTAIN          : {state_counts.get('ABSTAIN', 0)} "
          f"({100*state_counts.get('ABSTAIN',0)/total:.1f}%)")
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
