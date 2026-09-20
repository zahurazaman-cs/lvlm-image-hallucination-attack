# ============================================================
# mcnemar_test_all_defenses.py
# ============================================================
# Runs McNemar's exact test on every paired defense result across
# this entire project, using the "changed hallucinating -> correct"
# and "changed correct -> hallucinating" counts already recorded in
# each all-entries single-pass defense run's entry_state.json.
#
# McNemar's test is the correct test here because every comparison
# is PAIRED — the same entry, same image, before and after a single
# prompt change, not two independent samples. Only the entries that
# actually FLIPPED in either direction matter for the test; entries
# that stayed the same in both conditions carry no information about
# whether the change had a real effect.
#
#   b = flipped hallucinating -> correct (defense helped)
#   c = flipped correct -> hallucinating (defense hurt)
#
# Uses an exact two-sided binomial test on b out of (b+c) trials
# against p=0.5, which is the standard, correct form of McNemar's
# test for small paired samples (more reliable than the chi-square
# approximation when b+c is small, and it degrades gracefully to
# the same answer as sample sizes grow).
#
# THIS MAKES NO NEW MODEL CALLS — reads existing result files only.
# ============================================================

import json
import os
from math import comb


def exact_mcnemar_pvalue(b, c):
    """Exact two-sided binomial test for McNemar's test."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    # two-sided exact p-value: sum P(X<=k) + P(X>=n-k) under Binomial(n, 0.5)
    def binom_pmf(x, n):
        return comb(n, x) / (2 ** n)
    p_low = sum(binom_pmf(x, n) for x in range(0, k + 1))
    p_high = sum(binom_pmf(x, n) for x in range(n - k, n + 1))
    p_value = min(1.0, p_low + p_high)
    return p_value


def run_test(label, state_path, changed_field="changed",
             new_status_field="new_status", original_status_field="original_status"):
    if not os.path.exists(state_path):
        print(f"{label:<60s} FILE NOT FOUND, skipped")
        return None

    with open(state_path, "r") as f:
        state = json.load(f)

    b = 0  # hallucinating -> correct
    c = 0  # correct -> hallucinating
    for s in state.values():
        if not s.get(changed_field):
            continue
        new_status = s.get(new_status_field)
        orig_status = s.get(original_status_field)
        if new_status == "NONE" and orig_status in ("TARGETED", "UNTARGETED"):
            b += 1
        elif new_status in ("TARGETED", "UNTARGETED") and orig_status == "NONE":
            c += 1

    p_value = exact_mcnemar_pvalue(b, c)
    sig = ""
    if p_value is not None:
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."

    if p_value is None:
        print(f"{label:<60s} b=0, c=0 — no flips at all, test undefined")
    else:
        print(f"{label:<60s} b={b:<4d} c={c:<4d} p={p_value:.4f} {sig}")

    return {"label": label, "b": b, "c": c, "p_value": p_value}


print(f"{'Condition':<60s} {'b':<6s}{'c':<6s}{'p-value':<12s}")
print("-" * 95)

results = []

# ── qa_data.json (HaluEval) ───────────────────────────────────────
results.append(run_test(
    "qa_data.json, Gemma, Minimal Prompt Defense (all 500)",
    "outputs_step8_wrap_reclassify_all500_gemma/entry_state.json"))

results.append(run_test(
    "qa_data.json, Gemma, Structured Hall.-Analysis (all 500)",
    "outputs_step8_reclassify_all500_hallucination_analysis/entry_state.json"))

results.append(run_test(
    "qa_data.json, Qwen2.5-VL, Minimal Prompt Defense (all 500)",
    "outputs_step8_wrap_reclassify_all_qwen25/entry_state.json"))

results.append(run_test(
    "qa_data.json, Qwen2.5-VL, Structured Hall.-Analysis (all 500)",
    "outputs_step8_reclassify_all_qwen25_hallucination_analysis/entry_state.json"))

# ── NQ-Swap ────────────────────────────────────────────────────────
results.append(run_test(
    "NQ-Swap, Gemma, Minimal Prompt Defense (100)",
    "outputs_nqswap_100_wrap_reclassify/entry_state.json"))

results.append(run_test(
    "NQ-Swap, Gemma, Structured Hall.-Analysis (100)",
    "outputs_nqswap_100_hallucination_analysis/entry_state.json"))

results.append(run_test(
    "NQ-Swap, Qwen2.5-VL, Minimal Prompt Defense (100)",
    "outputs_qwen25_nqswap_100_wrap/entry_state.json"))

results.append(run_test(
    "NQ-Swap, Qwen2.5-VL, Structured Hall.-Analysis (100)",
    "outputs_qwen25_nqswap_100_hallucination_analysis/entry_state.json"))

# ── TruthfulQA ──────────────────────────────────────────────────────
results.append(run_test(
    "TruthfulQA, Gemma, Minimal Prompt Defense (22)",
    "outputs_truthfulqa_22_wrap_reclassify/entry_state.json"))

results.append(run_test(
    "TruthfulQA, Gemma, Structured Hall.-Analysis (22)",
    "outputs_truthfulqa_22_hallucination_analysis/entry_state.json"))

results.append(run_test(
    "TruthfulQA, Qwen2.5-VL, Minimal Prompt Defense (22)",
    "outputs_qwen25_truthfulqa_22_wrap/entry_state.json"))

results.append(run_test(
    "TruthfulQA, Qwen2.5-VL, Structured Hall.-Analysis (22)",
    "outputs_qwen25_truthfulqa_22_hallucination_analysis/entry_state.json"))

# ── Save full results ─────────────────────────────────────────────
valid_results = [r for r in results if r is not None]
with open("MCNEMAR_TEST_RESULTS.json", "w") as f:
    json.dump(valid_results, f, indent=2)

print()
print("Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant")
print("Saved full results to: MCNEMAR_TEST_RESULTS.json")
