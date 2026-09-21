# ============================================================
# filter_truthfulqa_person_entries.py
# ============================================================
# Downloads the full TruthfulQA dataset and filters it for
# entries where BOTH the correct answer and a specific incorrect
# answer look like real, named, individually photographable
# people — the shape this pipeline needs for its photo-based
# visual attack.
#
# SOLVES THE MISSING-CONTEXT PROBLEM: TruthfulQA has no dedicated
# knowledge/context field. Instead of scraping an external corpus,
# this pulls the longest sentence out of the semicolon-delimited
# "Correct Answers" field as the knowledge passage, since many
# entries already contain a full explanatory sentence there (for
# example, "Many people think X, but actually Y"), which reads
# exactly like the kind of documented true-fact passage your
# newspaper composition step needs.
#
# Heuristic for "looks like a person": 2-4 capitalized words, no
# digits — same approach used for the NQ-Swap filtering pass. This
# is a CANDIDATE list for manual review, not a final selection.
#
# Usage:
#   pip install datasets --break-system-packages
#   python3 filter_truthfulqa_person_entries.py
# ============================================================

import re
from datasets import load_dataset

OUTPUT_PATH = "truthfulqa_person_candidates.txt"

NAME_PATTERN = re.compile(r'^([A-Z][a-zA-Z\'\.-]*\s?){2,4}$')


def looks_like_person_name(answer):
    if not answer:
        return False
    answer = answer.strip()
    if any(ch.isdigit() for ch in answer):
        return False
    if not NAME_PATTERN.match(answer):
        return False
    word_count = len(answer.split())
    return 2 <= word_count <= 4


def best_knowledge_sentence(correct_answers_field):
    """Pick the longest sentence from the semicolon-delimited
    Correct Answers field — usually the most explanatory one."""
    parts = [p.strip() for p in correct_answers_field.split(";") if p.strip()]
    if not parts:
        return None
    return max(parts, key=len)


print("Loading TruthfulQA...")
dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
print(f"Total rows: {len(dataset)}")

candidates = []

for row in dataset:
    best_answer = row["Best Answer"]
    incorrect_answers = [a.strip() for a in row["Incorrect Answers"].split(";") if a.strip()]

    if not looks_like_person_name(best_answer):
        continue

    # Find the first incorrect answer that ALSO looks like a person name
    matching_incorrect = None
    for inc in incorrect_answers:
        if looks_like_person_name(inc):
            matching_incorrect = inc
            break

    if not matching_incorrect:
        continue

    knowledge = best_knowledge_sentence(row["Correct Answers"])
    if not knowledge or len(knowledge) < 20:
        continue

    candidates.append({
        "question": row["Question"],
        "best_answer": best_answer,
        "incorrect_answer": matching_incorrect,
        "knowledge": knowledge,
        "category": row["Category"],
    })

print(f"Candidates found (both answers look like person names): {len(candidates)}")

with open(OUTPUT_PATH, "w") as f:
    for i, c in enumerate(candidates):
        block = (
            f"{'='*70}\n"
            f"CANDIDATE #{i}  (Category: {c['category']})\n"
            f"{'='*70}\n"
            f"Question: {c['question']}\n"
            f"True Answer: {c['best_answer']}\n"
            f"Incorrect Answer: {c['incorrect_answer']}\n"
            f"Knowledge (from Correct Answers field): {c['knowledge']}\n\n"
        )
        f.write(block)

print(f"Saved candidate list to: {OUTPUT_PATH}")
print("Open that file, review the candidates, and pick whichever look "
      "cleanest — given you want this fast, even 15 to 20 entries is "
      "plenty for a full attack-plus-both-defenses run on both models.")
