# ============================================================
# make_gemma_vs_qwen_comparison_chart.py
# ============================================================
# Side-by-side comparison of hallucination classification on the
# SAME 51 entries (28 + 23), with only the VICTIM MODEL changed:
#   - Left pie:  Gemma-4-abliterated as victim (original results)
#   - Right pie: Qwen2.5-VL 7B as victim (victim-swap experiment)
#
# Reads:
#   Gemma : outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json
#           outputs_strict_batch_50/SUMMARY.json
#   Qwen  : outputs_victim_qwen_28/entry_state.json
#           outputs_victim_qwen_50/entry_state.json
#
# Saves to: outputs_victim_qwen_50/gemma_vs_qwen_comparison_chart.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

GEMMA_28_PATH = "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json"
GEMMA_50_PATH = "outputs_strict_batch_50/SUMMARY.json"
QWEN_28_PATH  = "outputs_victim_qwen_28/entry_state.json"
QWEN_50_PATH  = "outputs_victim_qwen_50/entry_state.json"
CHART_PATH    = "outputs_victim_qwen_50/gemma_vs_qwen_comparison_chart.png"


def load_gemma_28():
    with open(GEMMA_28_PATH, "r") as f:
        data = json.load(f)
    return [r["final_category"] for r in data]


def load_gemma_50():
    with open(GEMMA_50_PATH, "r") as f:
        data = json.load(f)
    return [r["status"] for r in data]


def load_qwen_state(path):
    with open(path, "r") as f:
        data = json.load(f)
    return [s["status"] for s in data.values()]


def counts(statuses):
    targeted   = sum(1 for c in statuses if c == "TARGETED")
    untargeted = sum(1 for c in statuses if c == "UNTARGETED")
    none_hall  = sum(1 for c in statuses if c == "NONE")
    scored     = targeted + untargeted + none_hall
    return targeted, untargeted, none_hall, scored


# ── Load both conditions ───────────────────────────────────────
gemma_statuses = load_gemma_28() + load_gemma_50()
qwen_statuses  = load_qwen_state(QWEN_28_PATH) + load_qwen_state(QWEN_50_PATH)

g_targeted, g_untargeted, g_none, g_scored = counts(gemma_statuses)
q_targeted, q_untargeted, q_none, q_scored = counts(qwen_statuses)

print(f"Gemma-4-abliterated victim ({g_scored} entries):")
print(f"  TARGETED   : {g_targeted} ({100*g_targeted/g_scored:.1f}%)")
print(f"  UNTARGETED : {g_untargeted} ({100*g_untargeted/g_scored:.1f}%)")
print(f"  NONE       : {g_none} ({100*g_none/g_scored:.1f}%)")
print(f"  Total hallucination: {g_targeted+g_untargeted} "
      f"({100*(g_targeted+g_untargeted)/g_scored:.1f}%)")
print()
print(f"Qwen2.5-VL 7B victim ({q_scored} entries):")
print(f"  TARGETED   : {q_targeted} ({100*q_targeted/q_scored:.1f}%)")
print(f"  UNTARGETED : {q_untargeted} ({100*q_untargeted/q_scored:.1f}%)")
print(f"  NONE       : {q_none} ({100*q_none/q_scored:.1f}%)")
print(f"  Total hallucination: {q_targeted+q_untargeted} "
      f"({100*(q_targeted+q_untargeted)/q_scored:.1f}%)")

# ── Build side-by-side pie charts ──────────────────────────────
colors = ["#e74c3c", "#f39c12", "#2ecc71"]
explode = (0.05, 0.05, 0.05)


def make_labels(targeted, untargeted, none_hall, scored):
    return [
        f"Targeted\n({targeted}, {100*targeted/scored:.1f}%)",
        f"Untargeted\n({untargeted}, {100*untargeted/scored:.1f}%)",
        f"None\n({none_hall}, {100*none_hall/scored:.1f}%)",
    ]


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left: Gemma
sizes_g = [g_targeted, g_untargeted, g_none]
labels_g = make_labels(g_targeted, g_untargeted, g_none, g_scored)
wedges_g, texts_g, autotexts_g = axes[0].pie(
    sizes_g, labels=labels_g, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts_g:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(12)
axes[0].set_title(
    f"Victim Model: Gemma-4-abliterated\n({g_scored} entries)",
    fontsize=13, fontweight="bold", pad=15
)
axes[0].axis("equal")

# Right: Qwen2.5-VL
sizes_q = [q_targeted, q_untargeted, q_none]
labels_q = make_labels(q_targeted, q_untargeted, q_none, q_scored)
wedges_q, texts_q, autotexts_q = axes[1].pie(
    sizes_q, labels=labels_q, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts_q:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(12)
axes[1].set_title(
    f"Victim Model: Qwen2.5-VL 7B\n({q_scored} entries)",
    fontsize=13, fontweight="bold", pad=15
)
axes[1].axis("equal")

fig.suptitle(
    "Hallucination Classification — Victim Model Comparison\n"
    f"Same {g_scored} entries, same pipeline — only the victim model changed",
    fontsize=15, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved comparison chart to: {CHART_PATH}")
