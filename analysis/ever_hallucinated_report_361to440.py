# ============================================================
# ever_hallucinated_report.py
# ============================================================
# Scans EVERY iteration's Step 9 classification file (not just the
# final one) across all three strict batches, and reports TWO
# statistics side by side:
#
#   1. FINAL-ITERATION statistic (what SUMMARY.json already reports)
#      — each entry's status is whatever its LAST iteration said.
#
#   2. "EVER HALLUCINATED" statistic — did this entry hallucinate
#      (targeted OR untargeted) on ANY of its iterations, even if
#      a later iteration happened to land on the correct answer?
#      This surfaces cases like Entry #200 (Millard Fillmore),
#      which hallucinated on 4/5 iterations but was recorded as
#      "NONE" because its 5th and final attempt got lucky.
#
# Reads iteration files from:
#   - outputs_strict_batch_28/entry_*/step9_reclassified_iterN.txt
#     (post-hoc reclassified, 3-way format)
#   - outputs_strict_batch_50/entry_*/step9_hallucination_iterN.txt
#     (native 3-way format)
#   - outputs_strict_batch_151_200/entry_*/step9_hallucination_iterN.txt
#     (native 3-way format)
#
# Writes:
#   EVER_HALLUCINATED_REPORT.txt (in the current directory)
# ============================================================

import glob
import json
import re

BATCHES = [
    {
        "name": "outputs_strict_batch_28",
        "iter_pattern": "step9_reclassified_iter*.txt",
        "summary_path": "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json",
        "final_category_key": "final_category",
    },
    {
        "name": "outputs_strict_batch_50",
        "iter_pattern": "step9_hallucination_iter*.txt",
        "summary_path": "outputs_strict_batch_50/SUMMARY.json",
        "final_category_key": "status",
    },
    {
        "name": "outputs_strict_batch_151_200",
        "iter_pattern": "step9_hallucination_iter*.txt",
        "summary_path": "outputs_strict_batch_151_200/SUMMARY.json",
        "final_category_key": "status",
    },
    {
        "name": "outputs_strict_batch_201_280",
        "iter_pattern": "step9_hallucination_iter*.txt",
        "summary_path": "outputs_strict_batch_201_280/SUMMARY.json",
        "final_category_key": "status",
    },
    {
        "name": "outputs_strict_batch_281_360",
        "iter_pattern": "step9_hallucination_iter*.txt",
        "summary_path": "outputs_strict_batch_281_360/SUMMARY.json",
        "final_category_key": "status",
    },
    {
        "name": "outputs_strict_batch_361_440_combined",
        "iter_pattern": "step9_hallucination_iter*.txt",
        "summary_path": "outputs_strict_batch_361_440_combined/SUMMARY.json",
        "final_category_key": "status",
    },
]


def classify_file(path):
    """Read a step9 classification file and return TARGETED / UNTARGETED / NONE."""
    with open(path, "r") as f:
        text = f.read()
    if "HALLUCINATING_TARGETED: YES" in text:
        return "TARGETED"
    elif "HALLUCINATING_UNTARGETED: YES" in text:
        return "UNTARGETED"
    else:
        return "NONE"


def load_final_statuses(batch):
    """Load each entry's officially-recorded FINAL status from its summary file."""
    with open(batch["summary_path"], "r") as f:
        data = json.load(f)
    key = batch["final_category_key"]
    return {r["entry_num"]: r[key] for r in data}


def scan_batch(batch):
    """
    For each entry folder in this batch, find all iteration files
    and classify every single one. Returns a dict:
      { entry_num: [category_iter1, category_iter2, ...] }
    """
    entry_dirs = sorted(
        glob.glob(f"{batch['name']}/entry_*"),
        key=lambda p: int(re.search(r"entry_(\d+)", p).group(1))
    )

    results = {}
    for entry_dir in entry_dirs:
        entry_num = int(re.search(r"entry_(\d+)", entry_dir).group(1))
        iter_files = sorted(
            glob.glob(f"{entry_dir}/{batch['iter_pattern']}"),
            key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
        )
        if not iter_files:
            continue  # e.g. SKIPPED_NO_PHOTO entries have no step9 files
        categories = [classify_file(f) for f in iter_files]
        results[entry_num] = categories
    return results


# ── Scan all three batches ─────────────────────────────────────
all_entries = {}  # entry_num -> (batch_name, [iteration categories])

for batch in BATCHES:
    scanned = scan_batch(batch)
    for entry_num, categories in scanned.items():
        all_entries[f"{batch['name']}#{entry_num}"] = {
            "batch": batch["name"],
            "entry_num": entry_num,
            "iterations": categories,
        }

total_scored = len(all_entries)

# ── Compute FINAL-ITERATION statistic (last iteration only) ───
final_targeted = sum(
    1 for e in all_entries.values() if e["iterations"][-1] == "TARGETED"
)
final_untargeted = sum(
    1 for e in all_entries.values() if e["iterations"][-1] == "UNTARGETED"
)
final_none = sum(
    1 for e in all_entries.values() if e["iterations"][-1] == "NONE"
)
final_any_hallucination = final_targeted + final_untargeted

# ── Compute EVER-HALLUCINATED statistic (any iteration) ────────
ever_targeted = sum(
    1 for e in all_entries.values() if "TARGETED" in e["iterations"]
)
ever_any_hallucination = sum(
    1 for e in all_entries.values()
    if "TARGETED" in e["iterations"] or "UNTARGETED" in e["iterations"]
)
never_hallucinated = sum(
    1 for e in all_entries.values()
    if all(c == "NONE" for c in e["iterations"])
)

# Entries that hallucinated at some point but the FINAL attempt
# happened to land on the correct answer (like Entry #200)
recovered_entries = [
    (key, e) for key, e in all_entries.items()
    if e["iterations"][-1] == "NONE"
    and ("TARGETED" in e["iterations"] or "UNTARGETED" in e["iterations"])
]

# ── Print + build report ────────────────────────────────────────
lines = []
lines.append("=" * 65)
lines.append("EVER-HALLUCINATED vs FINAL-ITERATION REPORT")
lines.append("=" * 65)
lines.append(f"Total entries scored (across all 3 batches): {total_scored}")
lines.append("")
lines.append("-" * 65)
lines.append("STATISTIC 1: FINAL ITERATION ONLY (what SUMMARY.json reports)")
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
lines.append(f"  EVER hit TARGETED (at least once)       : {ever_targeted} "
              f"({100*ever_targeted/total_scored:.1f}%)")
lines.append(f"  EVER hallucinated, targeted OR untargeted: "
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
    f"but 'recovered' to the correct answer by their final iteration "
    f"(masked in the final-only statistic)."
)
lines.append("")

if recovered_entries:
    lines.append("-" * 65)
    lines.append("ENTRIES THAT HALLUCINATED THEN 'RECOVERED' BY FINAL ITERATION:")
    lines.append("-" * 65)
    for key, e in recovered_entries:
        lines.append(
            f"  {e['batch']} / Entry #{e['entry_num']}: "
            f"iterations = {e['iterations']}"
        )
    lines.append("")

lines.append("-" * 65)
lines.append("PER-ENTRY DETAIL (all iterations)")
lines.append("-" * 65)
for key, e in sorted(all_entries.items(),
                      key=lambda kv: (kv[1]["batch"], kv[1]["entry_num"])):
    ever_hall = ("TARGETED" in e["iterations"]
                 or "UNTARGETED" in e["iterations"])
    lines.append(
        f"  {e['batch']:28s} Entry #{e['entry_num']:3d}: "
        f"final={e['iterations'][-1]:10s} | "
        f"ever_hallucinated={'YES' if ever_hall else 'NO ':3s} | "
        f"all_iterations={e['iterations']}"
    )

report_text = "\n".join(lines)
print(report_text)

with open("EVER_HALLUCINATED_REPORT.txt", "w") as f:
    f.write(report_text)

print("\nSaved report to: EVER_HALLUCINATED_REPORT.txt")
