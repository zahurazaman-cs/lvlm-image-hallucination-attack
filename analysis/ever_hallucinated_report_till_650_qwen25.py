# ============================================================
# ever_hallucinated_report_till_650_qwen25.py
# ============================================================
# Scans EVERY iteration's Step 9 classification file (not just the
# final one) across ALL THREE Qwen2.5-VL victim-model batches:
#   - outputs_victim_qwen_28        (28 entries)
#   - outputs_victim_qwen_50        (23 entries)
#   - outputs_victim_qwen_601_650   (30 entries)
#
# Victim model (target model, Step 8): Qwen2.5-VL 7B (via Ollama)
# Judge model (Step 9 classifier): Gemma-4-abliterated
#
# Reports TWO statistics side by side:
#   1. FINAL-ITERATION statistic — each entry's status is whatever
#      its LAST iteration said (what entry_state.json reports).
#   2. "EVER HALLUCINATED" statistic — did this entry hallucinate
#      (targeted OR untargeted) on ANY of its iterations, even if a
#      later iteration happened to land on the correct answer?
#
# Writes: Ever_hallucinated_report_qwen25.txt
# ============================================================

import glob
import re

BATCH_DIRS = [
    "outputs_victim_qwen_28",
    "outputs_victim_qwen_50",
    "outputs_victim_qwen_601_650",
]
REPORT_PATH = "Ever_hallucinated_report_qwen25.txt"


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

for batch_dir in BATCH_DIRS:
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
              "outputs_victim_qwen_50 + outputs_victim_qwen_601_650")
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
    f"at least once during escalation."
)
lines.append(
    f"  {len(recovered_entries)} entries hallucinated at some point "
    f"but 'recovered' to the correct answer by their final iteration."
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
