# ============================================================
# transferability_analysis_full500_gemma_qwen25.py
# ============================================================
# FULL 500-entry cross-model hallucination transferability
# analysis on qa_data.json (HaluEval), Gemma-4-abliterated vs
# Qwen2.5-VL. Combines:
#
#   - The 449 reuse-method entries (same image shown to both
#     models by construction — no new calls needed)
#   - The 51 two-phase entries, using the two new cross tests:
#       crosstest_qwen25_on_gemma_images_twophase.py
#       crosstest_gemma_on_qwen25_images_twophase.py
#
# This is a PURE ANALYSIS script — it makes no model calls itself,
# it only reads already-computed results, including the two cross
# test result files above (run those two scripts FIRST).
#
# DIRECTION 1 — of ALL 332 entries Gemma hallucinated on (the
# complete set across all 14 batches), what fraction also fooled
# Qwen2.5-VL on that SAME image?
#
# DIRECTION 2 — of ALL 107 entries Qwen2.5-VL hallucinated on (the
# complete set across all 14 batches, both two-phase and
# reuse-method), what fraction also fooled Gemma on that SAME
# image?
#
# Judge model throughout: Gemma-4-abliterated, unchanged.
# ============================================================

import os
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

CROSSTEST_QWEN_ON_GEMMA_PATH = "outputs_crosstest_qwen25_on_gemma_images_twophase/entry_state.json"
CROSSTEST_GEMMA_ON_QWEN_PATH = "outputs_crosstest_gemma_on_qwen25_images_twophase/entry_state.json"
FILL_MISSING_PATH = "outputs_fill_missing_qwen_on_gemma_images/entry_state.json"

OUTPUT_REPORT_PATH = "TRANSFERABILITY_ANALYSIS_FULL500_gemma_qwen25.txt"


def load_gemma_batch(path):
    with open(f"{path}/SUMMARY.json", "r") as f:
        return {r["entry_num"]: r["status"] for r in json.load(f)}


def load_qwen_batch(path):
    with open(f"{path}/entry_state.json", "r") as f:
        state = json.load(f)
    return {s["entry_num"]: s["status"] for s in state.values()}


# ── Build the reuse-method (Gemma label, Qwen label) pairs (449) ──
pairs = []  # (batch_name, entry_num, gemma_label, qwen_label)

for qwen_dir, gemma_dir in REUSE_BATCH_PAIRS:
    gemma_labels = load_gemma_batch(gemma_dir)
    qwen_labels  = load_qwen_batch(qwen_dir)
    shared = set(gemma_labels) & set(qwen_labels)
    for entry_num in sorted(shared):
        g_label = gemma_labels[entry_num]
        q_label = qwen_labels[entry_num]
        if g_label == "SKIPPED_NO_PHOTO":
            continue
        pairs.append((gemma_dir, entry_num, g_label, q_label))

reuse_count = len(pairs)

# ── Add the two-phase pairs from the two cross tests (up to 51) ──
with open(CROSSTEST_QWEN_ON_GEMMA_PATH, "r") as f:
    crosstest_qog = json.load(f)  # keyed by "batch::entry", has gemma_status + status (Qwen's)

with open(CROSSTEST_GEMMA_ON_QWEN_PATH, "r") as f:
    crosstest_goq = json.load(f)  # keyed by "batch::entry", has qwen_status + status (Gemma's)

# crosstest_qog gives us: gemma_status (Gemma's ORIGINAL label on its own
# image) + status (Qwen's label on THAT SAME Gemma image) -> valid pair
for key, s in crosstest_qog.items():
    if s.get("status") == "ERROR" or "gemma_status" not in s:
        continue
    pairs.append((s["batch_dir"], s["entry_num"], s["gemma_status"], s["status"]))

# crosstest_goq gives us: qwen_status (Qwen's ORIGINAL label on its own
# image) + status (Gemma's label on THAT SAME Qwen image) -> valid pair,
# but note this is the OTHER direction's image, so it must be tracked
# separately rather than merged into the same (gemma_img) pairs list.
twophase_qwen_img_pairs = []  # (batch_name, entry_num, gemma_label_on_qwen_img, qwen_original_label)
for key, s in crosstest_goq.items():
    if s.get("status") == "ERROR" or "qwen_status" not in s:
        continue
    twophase_qwen_img_pairs.append(
        (s["batch_dir"], s["entry_num"], s["status"], s["qwen_status"])
    )

# ── Add the fill-in pairs, if that script has been run ───────────
if os.path.exists(FILL_MISSING_PATH):
    with open(FILL_MISSING_PATH, "r") as f:
        fill_state = json.load(f)
    for key, s in fill_state.items():
        if s.get("status") == "ERROR" or "gemma_status" not in s:
            continue
        pairs.append((s["batch_dir"], s["entry_num"], s["gemma_status"], s["status"]))

total_gemma_img_pairs = len(pairs)  # 449 reuse + up to 51 two-phase + fill-ins = up to 500

print(f"Gemma-image pairs available (for Direction 1): {total_gemma_img_pairs}")
print(f"  ({reuse_count} from reuse-method batches + "
      f"{total_gemma_img_pairs - reuse_count} from two-phase cross test)")
print(f"Qwen-image pairs available (for Direction 2, two-phase only): "
      f"{len(twophase_qwen_img_pairs)}")
print()

# ============================================================
# DIRECTION 1: ALL Gemma-hallucinated entries -> Qwen outcome
# ============================================================
# Uses: reuse-method pairs (Gemma's image, both labels already
# known) + two-phase pairs from crosstest_qog (Gemma's own image,
# Qwen's label on that same image).

gemma_hallucinated = [p for p in pairs if p[2] in ("TARGETED", "UNTARGETED")]

d1_tt = sum(1 for p in gemma_hallucinated if p[3] == "TARGETED")
d1_tu = sum(1 for p in gemma_hallucinated if p[3] == "UNTARGETED")
d1_nt = sum(1 for p in gemma_hallucinated if p[3] == "NONE")
d1_total = len(gemma_hallucinated)
d1_transferred = d1_tt + d1_tu

print("=" * 70)
print("DIRECTION 1: ALL Gemma-hallucinated entries -> Qwen2.5-VL outcome")
print("=" * 70)
print(f"Total Gemma-hallucinated entries covered: {d1_total} (of 332 total)")
if d1_total:
    print(f"  Transferred (Qwen also TARGETED)   : {d1_tt} ({100*d1_tt/d1_total:.1f}%)")
    print(f"  Transferred (Qwen UNTARGETED)       : {d1_tu} ({100*d1_tu/d1_total:.1f}%)")
    print(f"  Total transferred                   : {d1_transferred} "
          f"({100*d1_transferred/d1_total:.1f}%)")
    print(f"  Did NOT transfer (Qwen correct)     : {d1_nt} ({100*d1_nt/d1_total:.1f}%)")
if d1_total < 332:
    print(f"  NOTE: {332 - d1_total} of the 332 Gemma-hallucinated entries "
          f"are still missing — check that both cross test scripts "
          f"completed without errors.")

# ============================================================
# DIRECTION 2: ALL Qwen-hallucinated entries -> Gemma outcome
# ============================================================
# Uses: reuse-method pairs (same image, Gemma's original label IS
# the answer) + two-phase pairs from crosstest_goq (Qwen's own
# image, Gemma's label on that same image).

qwen_hallucinated_reuse = [p for p in pairs[:reuse_count] if p[3] in ("TARGETED", "UNTARGETED")]
qwen_hallucinated_twophase = [
    p for p in twophase_qwen_img_pairs if p[3] in ("TARGETED", "UNTARGETED")
]

d2_tt = (sum(1 for p in qwen_hallucinated_reuse if p[2] == "TARGETED") +
         sum(1 for p in qwen_hallucinated_twophase if p[2] == "TARGETED"))
d2_tu = (sum(1 for p in qwen_hallucinated_reuse if p[2] == "UNTARGETED") +
         sum(1 for p in qwen_hallucinated_twophase if p[2] == "UNTARGETED"))
d2_nt = (sum(1 for p in qwen_hallucinated_reuse if p[2] == "NONE") +
         sum(1 for p in qwen_hallucinated_twophase if p[2] == "NONE"))
d2_total = len(qwen_hallucinated_reuse) + len(qwen_hallucinated_twophase)
d2_transferred = d2_tt + d2_tu

print()
print("=" * 70)
print("DIRECTION 2: ALL Qwen2.5-VL-hallucinated entries -> Gemma outcome")
print("=" * 70)
print(f"Total Qwen-hallucinated entries covered: {d2_total} (of 107 total)")
if d2_total:
    print(f"  Transferred (Gemma also TARGETED)  : {d2_tt} ({100*d2_tt/d2_total:.1f}%)")
    print(f"  Transferred (Gemma UNTARGETED)      : {d2_tu} ({100*d2_tu/d2_total:.1f}%)")
    print(f"  Total transferred                   : {d2_transferred} "
          f"({100*d2_transferred/d2_total:.1f}%)")
    print(f"  Did NOT transfer (Gemma correct)    : {d2_nt} ({100*d2_nt/d2_total:.1f}%)")
if d2_total < 107:
    print(f"  NOTE: {107 - d2_total} of the 107 Qwen-hallucinated entries "
          f"are still missing — check that both cross test scripts "
          f"completed without errors.")

with open(OUTPUT_REPORT_PATH, "w") as f:
    f.write("FULL 500-ENTRY TRANSFERABILITY ANALYSIS — Gemma vs Qwen2.5-VL\n\n")
    f.write(f"Direction 1 (Gemma->Qwen): {d1_total}/332 entries covered, "
            f"{100*d1_transferred/d1_total:.1f}% transferred\n" if d1_total else "")
    f.write(f"Direction 2 (Qwen->Gemma): {d2_total}/107 entries covered, "
            f"{100*d2_transferred/d2_total:.1f}% transferred\n" if d2_total else "")

print(f"\nSummary saved to: {OUTPUT_REPORT_PATH}")
