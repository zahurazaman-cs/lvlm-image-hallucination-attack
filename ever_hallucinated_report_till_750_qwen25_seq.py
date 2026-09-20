# ============================================================
# ever_hallucinated_report_till_750_qwen25_seq.py
# ============================================================
# Combines all ELEVEN Qwen2.5-VL victim-model batches:
#   - outputs_victim_qwen_28              (28 entries, two-phase
#     pipeline — has MULTIPLE iterations per entry via step9_
#     hallucination_iterN.txt files)
#   - outputs_victim_qwen_50              (23 entries, two-phase
#     pipeline — same multi-iteration structure)
#   - outputs_victim_qwen_151_200         (26 entries, reuse method
#     — only ONE pass per entry, step9_hallucination.txt)
#   - outputs_victim_qwen_201_280         (48 entries, reuse method)
#   - outputs_victim_qwen_281_360         (50 entries, reuse method)
#   - outputs_victim_qwen_361_440         (50 entries, reuse method)
#   - outputs_victim_qwen_441_520         (49 entries, reuse method)
#   - outputs_victim_qwen_521_600         (47 entries, reuse method)
#   - outputs_victim_qwen_601_650_reuse   (30 entries, reuse method;
#     NOTE this is a DIFFERENT, distinct folder from the earlier
#     abandoned two-phase attempt at this same batch range)
#   - outputs_victim_qwen_651_700         (30 entries, reuse method)
#   - outputs_victim_qwen_701_750         (30 entries, reuse method)
#
# For the two two-phase batches, reports the full "ever hallucinated
# during escalation vs final iteration" comparison, same as your
# Gemma-victim reports. For the nine reuse-method batches, there is
# only one iteration per entry by design (the final Gemma-escalated
# image was reused directly rather than re-escalated), so "final"
# and "ever" are identical for those entries — this is noted
# explicitly rather than treated as a coincidence.
#
# Writes: Ever_hallucinated_report_till_750_qwen25_seq.txt
# ============================================================

import glob
import re
import json

TWO_PHASE_BATCH_DIRS = [
    "outputs_victim_qwen_28",
    "outputs_victim_qwen_50",
]
REUSE_BATCH_DIRS = [
    "outputs_victim_qwen_151_200",
    "outputs_victim_qwen_201_280",
    "outputs_victim_qwen_281_360",
    "outputs_victim_qwen_361_440",
    "outputs_victim_qwen_441_520",
    "outputs_victim_qwen_521_600",
    "outputs_victim_qwen_601_650_reuse",
    "outputs_victim_qwen_651_700",
    "outputs_victim_qwen_701_750",
]
REPORT_PATH = "Ever_hallucinated_report_till_750_qwen25_seq.txt"


def classify_file(path):
    with open(path, "r") as f:
        text = f.read()
    if "HALLUCINATING_TARGETED: YES" in text:
        return "TARGETED"
    elif "HALLUCINATING_UNTARGETED: YES" in text:
        return "UNTARGETED"
    else:
        return "NONE"


all_entries = {}  # (batch_name, entry_num) -> [iteration categories]

# ── Two-phase batches: full iteration history available ────────
for batch_dir in TWO_PHASE_BATCH_DIRS:
    entry_dirs = sorted(
        glob.glob(f"{batch_dir}/entry_[0-9]*"),
        key=lambda p: int(re.search(r"entry_(\d+)", p).group(1))
    )
    for entry_dir in entry_dirs:
        entry_num = int(re.search(r"entry_(\d+)", entry_dir).group(1))
        iter_files = sorted(
            glob.glob(f"{entry_dir}/step9_hallucination_iter*.txt"),
            key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
        )
        if not iter_files:
            continue  # SKIPPED_NO_PHOTO entries have no step9 files
        categories = [classify_file(f) for f in iter_files]
        all_entries[(batch_dir, entry_num)] = categories

# ── Reuse-method batches: single pass per entry ─────────────────
for batch_dir in REUSE_BATCH_DIRS:
    with open(f"{batch_dir}/entry_state.json", "r") as f:
        reuse_state = json.load(f)
    for key, s in reuse_state.items():
        if s["status"] in ("SKIPPED_NO_PHOTO", "ERROR"):
            continue
        all_entries[(batch_dir, s["entry_num"])] = [s["status"]]

total_scored = len(all_entries)

# ── FINAL-ITERATION statistic ──────────────────────────────────
final_targeted = sum(1 for c in all_entries.values() if c[-1] == "TARGETED")
final_untargeted = sum(1 for c in all_entries.values() if c[-1] == "UNTARGETED")
final_none = sum(1 for c in all_entries.values() if c[-1] == "NONE")
final_any_hallucination = final_targeted + final_untargeted

# ── EVER-HALLUCINATED statistic ────────────────────────────────
ever_targeted = sum(1 for c in all_entries.values() if "TARGETED" in c)
ever_any_hallucination = sum(
    1 for c in all_entries.values()
    if "TARGETED" in c or "UNTARGETED" in c
)
never_hallucinated = sum(
    1 for c in all_entries.values() if all(x == "NONE" for x in c)
)

recovered_entries = [
    (key, c) for key, c in all_entries.items()
    if c[-1] == "NONE" and ("TARGETED" in c or "UNTARGETED" in c)
]

# ── Build report ────────────────────────────────────────────────
lines = []
lines.append("=" * 65)
lines.append("EVER-HALLUCINATED vs FINAL-ITERATION REPORT")
lines.append("Victim model: Qwen2.5-VL 7B  |  Judge: Gemma-4-abliterated")
lines.append("Combined across outputs_victim_qwen_28 + "
              "outputs_victim_qwen_50 +")
lines.append("outputs_victim_qwen_151_200 + outputs_victim_qwen_201_280 + "
              "outputs_victim_qwen_281_360 +")
lines.append("outputs_victim_qwen_361_440 + outputs_victim_qwen_441_520 + "
              "outputs_victim_qwen_521_600 +")
lines.append("outputs_victim_qwen_601_650_reuse + outputs_victim_qwen_651_700 + "
              "outputs_victim_qwen_701_750")
lines.append("")
lines.append("NOTE: outputs_victim_qwen_151_200, outputs_victim_qwen_")
lines.append("201_280, outputs_victim_qwen_281_360, outputs_victim_")
lines.append("qwen_361_440, outputs_victim_qwen_441_520, outputs_")
lines.append("victim_qwen_521_600, outputs_victim_qwen_601_650_reuse,")
lines.append("outputs_victim_qwen_651_700, and outputs_victim_qwen_701_750")
lines.append("used the 'reuse' method — Steps 1-7 images/")
lines.append("newspapers were taken directly from the Gemma-victim run's")
lines.append("FINAL iteration and reused as-is; only Steps 8/9 were")
lines.append("rerun with Qwen2.5-VL as victim. These entries therefore")
lines.append("have exactly ONE pass each, so their 'final' and 'ever")
lines.append("hallucinated' values are identical by construction, not")
lines.append("because escalation happened to stop early.")
lines.append("=" * 65)
lines.append(f"Total entries scored: {total_scored}")
lines.append("")
lines.append("-" * 65)
lines.append("STATISTIC 1: FINAL ITERATION ONLY")
lines.append("-" * 65)
lines.append(f"  TARGETED hallucination   : {final_targeted} "
              f"({100*final_targeted/total_scored:.1f}%)")
lines.append(f"  UNTARGETED hallucination : {final_untargeted} "
              f"({100*final_untargeted/total_scored:.1f}%)")
lines.append(f"  NOT hallucinating        : {final_none} "
              f"({100*final_none/total_scored:.1f}%)")
lines.append(f"  TOTAL hallucination (final attempt) : "
              f"{final_any_hallucination} "
              f"({100*final_any_hallucination/total_scored:.1f}%)")
lines.append("")
lines.append("-" * 65)
lines.append("STATISTIC 2: EVER HALLUCINATED (any iteration during escalation)")
lines.append("-" * 65)
lines.append(f"  EVER hit TARGETED (at least once)        : {ever_targeted} "
              f"({100*ever_targeted/total_scored:.1f}%)")
lines.append(f"  EVER hallucinated, targeted OR untargeted : "
              f"{ever_any_hallucination} "
              f"({100*ever_any_hallucination/total_scored:.1f}%)")
lines.append(f"  NEVER hallucinated (all iterations correct): "
              f"{never_hallucinated} "
              f"({100*never_hallucinated/total_scored:.1f}%)")
lines.append("")
lines.append("-" * 65)
lines.append("SUMMARY COMPARISON")
lines.append("-" * 65)
lines.append(
    f"  {100*final_any_hallucination/total_scored:.1f}% of entries "
    f"hallucinated on their FINAL attempt, but "
    f"{100*ever_any_hallucination/total_scored:.1f}% hallucinated "
    f"at least once during escalation (where escalation applies)."
)
lines.append(
    f"  {len(recovered_entries)} entries hallucinated at some point "
    f"but 'recovered' to the correct answer by their final iteration "
    f"(only possible for the two two-phase batches)."
)
lines.append("")

if recovered_entries:
    lines.append("-" * 65)
    lines.append("ENTRIES THAT HALLUCINATED THEN 'RECOVERED' BY FINAL ITERATION:")
    lines.append("-" * 65)
    for (batch_dir, num), c in recovered_entries:
        lines.append(f"  {batch_dir} / Entry #{num}: iterations = {c}")
    lines.append("")

lines.append("-" * 65)
lines.append("PER-ENTRY DETAIL (all iterations)")
lines.append("-" * 65)
for (batch_dir, num), c in sorted(all_entries.items(),
                                   key=lambda kv: (kv[0][0], kv[0][1])):
    ever_hall = "TARGETED" in c or "UNTARGETED" in c
    lines.append(
        f"  {batch_dir:26s} Entry #{num:3d}: "
        f"final={c[-1]:10s} | "
        f"ever_hallucinated={'YES' if ever_hall else 'NO ':3s} | "
        f"all_iterations={c}"
    )

report_text = "\n".join(lines)
print(report_text)

with open(REPORT_PATH, "w") as f:
    f.write(report_text)

print(f"\nSaved report to: {REPORT_PATH}")
