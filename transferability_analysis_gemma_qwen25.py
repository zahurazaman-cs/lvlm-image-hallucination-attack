# ============================================================
# transferability_analysis_gemma_qwen25.py
# ============================================================
# Cross-model hallucination transferability analysis on
# qa_data.json (HaluEval), Gemma-4-abliterated vs Qwen2.5-VL.
#
# This is a PURE ANALYSIS script — it makes no model calls and
# regenerates nothing. It only reads already-computed results.
#
# SCOPE: restricted to the 449 entries where the Qwen2.5-VL victim
# run REUSED the exact same adversarial image that Gemma was
# originally tested against (the twelve reuse-method batches).
# This is the only subset where a same-image, different-model
# comparison is methodologically valid.
#
# EXCLUDED: the 51 entries from the two earliest batches
# (outputs_victim_qwen_28, outputs_victim_qwen_50), where a FRESH
# image was generated and escalated specifically for Qwen2.5-VL,
# separate from whichever image fooled Gemma on those same
# questions. Comparing labels there would confound "different
# model" with "different image," so they are not included in this
# transferability claim. If you want, a small follow-up run could
# test each model on the OTHER model's image for these 51 entries
# specifically, closing this gap.
#
# Judge model throughout: Gemma-4-abliterated, unchanged.
#
# Two directions reported, both drawn from the SAME underlying
# 449-entry table of (Gemma label, Qwen label) pairs, just
# filtered from opposite starting points:
#
#   DIRECTION 1 — starts from Gemma's hallucinated entries, checks
#   what Qwen2.5-VL did on that same image.
#
#   DIRECTION 2 — starts from Qwen2.5-VL's hallucinated entries
#   (restricted to this same 449-entry subset), checks what Gemma
#   did on that same image.
#
# For every entry, the SPECIFIC hallucination category (TARGETED
# or UNTARGETED) is recorded for both models, not just a binary
# hallucinated/did-not-hallucinate flag.
#
# Outputs a full per-entry breakdown plus summary tables for both
# directions.
# ============================================================

import json

REUSE_BATCH_PAIRS = [
    ("outputs_victim_qwen_151_200",       "outputs_strict_batch_151_200"),
    ("outputs_victim_qwen_201_280",       "outputs_strict_batch_201_280"),
    ("outputs_victim_qwen_281_360",       "outputs_strict_batch_281_360"),
    ("outputs_victim_qwen_361_440",       "outputs_strict_batch_361_440_combined"),
    ("outputs_victim_qwen_441_520",       "outputs_strict_batch_441_520"),
    ("outputs_victim_qwen_521_600",       "outputs_strict_batch_521_600"),
    ("outputs_victim_qwen_601_650_reuse", "outputs_strict_batch_601_650"),
    ("outputs_victim_qwen_651_700",       "outputs_strict_batch_651_700"),
    ("outputs_victim_qwen_701_750",       "outputs_strict_batch_701_750"),
    ("outputs_victim_qwen_751_800",       "outputs_strict_batch_751_800"),
    ("outputs_victim_qwen_801_850",       "outputs_strict_batch_801_850"),
    ("outputs_victim_qwen_851_900",       "outputs_strict_batch_851_900"),
]

OUTPUT_REPORT_PATH = "TRANSFERABILITY_ANALYSIS_gemma_qwen25.txt"


def load_gemma_batch(path):
    with open(f"{path}/SUMMARY.json", "r") as f:
        return {r["entry_num"]: r["status"] for r in json.load(f)}


def load_qwen_batch(path):
    with open(f"{path}/entry_state.json", "r") as f:
        state = json.load(f)
    return {s["entry_num"]: s["status"] for s in state.values()}


# ── Build the single 449-entry (Gemma label, Qwen label) table ──
pairs = []  # list of (batch_pair_name, entry_num, gemma_label, qwen_label)

for qwen_dir, gemma_dir in REUSE_BATCH_PAIRS:
    gemma_labels = load_gemma_batch(gemma_dir)
    qwen_labels  = load_qwen_batch(qwen_dir)

    shared_entries = set(gemma_labels) & set(qwen_labels)
    for entry_num in sorted(shared_entries):
        g_label = gemma_labels[entry_num]
        q_label = qwen_labels[entry_num]
        if g_label == "SKIPPED_NO_PHOTO":
            continue
        pairs.append((gemma_dir, entry_num, g_label, q_label))

print(f"Total shared same-image entries analyzed: {len(pairs)}")
print(f"(Excludes the 51 two-phase entries where Qwen was tested "
      f"on a separately-generated image — see script header.)")
print()

# ============================================================
# DIRECTION 1: Gemma hallucinated -> what did Qwen do?
# ============================================================

gemma_hallucinated = [
    p for p in pairs if p[2] in ("TARGETED", "UNTARGETED")
]

d1_transferred_targeted   = 0  # Gemma hall. -> Qwen also TARGETED
d1_transferred_untargeted = 0  # Gemma hall. -> Qwen also UNTARGETED (any kind)
d1_did_not_transfer       = 0  # Gemma hall. -> Qwen answered correctly (NONE)

d1_detail = []
for batch, entry_num, g_label, q_label in gemma_hallucinated:
    if q_label == "TARGETED":
        d1_transferred_targeted += 1
        outcome = "TRANSFERRED (Qwen: TARGETED)"
    elif q_label == "UNTARGETED":
        d1_transferred_untargeted += 1
        outcome = "TRANSFERRED (Qwen: UNTARGETED)"
    else:
        d1_did_not_transfer += 1
        outcome = "DID NOT TRANSFER (Qwen: NONE)"
    d1_detail.append(
        f"{batch} Entry #{entry_num}: Gemma={g_label:10s} -> Qwen={q_label:10s}  [{outcome}]"
    )

d1_total = len(gemma_hallucinated)
d1_transferred_total = d1_transferred_targeted + d1_transferred_untargeted

print("=" * 70)
print("DIRECTION 1: Gemma-hallucinated entries -> Qwen2.5-VL outcome")
print("=" * 70)
print(f"Total Gemma-hallucinated entries in this subset: {d1_total}")
if d1_total:
    print(f"  Transferred (Qwen also TARGETED)  : {d1_transferred_targeted} "
          f"({100*d1_transferred_targeted/d1_total:.1f}%)")
    print(f"  Transferred (Qwen UNTARGETED)      : {d1_transferred_untargeted} "
          f"({100*d1_transferred_untargeted/d1_total:.1f}%)")
    print(f"  Total transferred (any hallucination): {d1_transferred_total} "
          f"({100*d1_transferred_total/d1_total:.1f}%)")
    print(f"  Did NOT transfer (Qwen correct)    : {d1_did_not_transfer} "
          f"({100*d1_did_not_transfer/d1_total:.1f}%)")

# ============================================================
# DIRECTION 2: Qwen hallucinated (within this subset) -> what did Gemma do?
# ============================================================
# Note: since these are the SAME images, Gemma's label here IS its
# ORIGINAL attack result on that image — no new computation needed,
# just filtering the same table from the opposite direction.

qwen_hallucinated = [
    p for p in pairs if p[3] in ("TARGETED", "UNTARGETED")
]

d2_transferred_targeted   = 0  # Qwen hall. -> Gemma also TARGETED
d2_transferred_untargeted = 0  # Qwen hall. -> Gemma also UNTARGETED
d2_did_not_transfer       = 0  # Qwen hall. -> Gemma answered correctly (NONE)

d2_detail = []
for batch, entry_num, g_label, q_label in qwen_hallucinated:
    if g_label == "TARGETED":
        d2_transferred_targeted += 1
        outcome = "TRANSFERRED (Gemma: TARGETED)"
    elif g_label == "UNTARGETED":
        d2_transferred_untargeted += 1
        outcome = "TRANSFERRED (Gemma: UNTARGETED)"
    else:
        d2_did_not_transfer += 1
        outcome = "DID NOT TRANSFER (Gemma: NONE)"
    d2_detail.append(
        f"{batch} Entry #{entry_num}: Qwen={q_label:10s} -> Gemma={g_label:10s}  [{outcome}]"
    )

d2_total = len(qwen_hallucinated)
d2_transferred_total = d2_transferred_targeted + d2_transferred_untargeted

print()
print("=" * 70)
print("DIRECTION 2: Qwen2.5-VL-hallucinated entries -> Gemma outcome")
print("=" * 70)
print(f"Total Qwen-hallucinated entries in this subset: {d2_total}")
if d2_total:
    print(f"  Transferred (Gemma also TARGETED) : {d2_transferred_targeted} "
          f"({100*d2_transferred_targeted/d2_total:.1f}%)")
    print(f"  Transferred (Gemma UNTARGETED)     : {d2_transferred_untargeted} "
          f"({100*d2_transferred_untargeted/d2_total:.1f}%)")
    print(f"  Total transferred (any hallucination): {d2_transferred_total} "
          f"({100*d2_transferred_total/d2_total:.1f}%)")
    print(f"  Did NOT transfer (Gemma correct)   : {d2_did_not_transfer} "
          f"({100*d2_did_not_transfer/d2_total:.1f}%)")

# ── Save full detail report ───────────────────────────────────────
with open(OUTPUT_REPORT_PATH, "w") as f:
    f.write("TRANSFERABILITY ANALYSIS — Gemma vs Qwen2.5-VL (HaluEval)\n")
    f.write(f"Subset: {len(pairs)} entries tested on identical images\n")
    f.write("(51 two-phase entries excluded — see script header)\n\n")

    f.write("=" * 70 + "\n")
    f.write("DIRECTION 1 DETAIL: Gemma hallucinated -> Qwen outcome\n")
    f.write("=" * 70 + "\n")
    f.write("\n".join(d1_detail))

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("DIRECTION 2 DETAIL: Qwen hallucinated -> Gemma outcome\n")
    f.write("=" * 70 + "\n")
    f.write("\n".join(d2_detail))

print(f"\nFull per-entry detail saved to: {OUTPUT_REPORT_PATH}")
