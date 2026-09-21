# ============================================================
# filter_truthfulqa_person_entries_v2.py
# ============================================================
# Looser version of the TruthfulQA person filter. The first
# version required the ENTIRE answer field to be nothing but a
# 2-4 word capitalized name, which rejected perfectly good
# candidates where the name is embedded in a longer sentence, for
# example "Sherlock Holmes and Dr. Watson live on Baker Street"
# was rejected outright even though "Sherlock Holmes" is right
# there.
#
# This version searches for a name-shaped substring ANYWHERE
# inside the Best Answer and each Incorrect Answer, rather than
# requiring the whole field to match. This will surface more
# false positives (place names, fictional titles) alongside real
# candidates, since this remains a review list, not a final
# selection — you pick the clean ones out by hand afterward, same
# as with NQ-Swap.
#
# Same knowledge-extraction approach as before: pulls the longest
# sentence out of the semicolon-delimited Correct Answers field.
#
# Usage:
#   python3 filter_truthfulqa_person_entries_v2.py
# ============================================================

import re
from datasets import load_dataset

OUTPUT_PATH = "truthfulqa_person_candidates_v2.txt"

# Matches a 2-4 word capitalized name-shaped phrase ANYWHERE in a string
NAME_SUBSTRING_PATTERN = re.compile(
    r'\b([A-Z][a-zA-Z\'\.]+(?:\s+[A-Z][a-zA-Z\'\.]+){1,3})\b'
)

# Common false-positive starters to quietly skip (place names, titles,
# generic capitalized phrases that aren't people) — not exhaustive,
# just cuts obvious noise before you review by hand
SKIP_IF_STARTS_WITH = (
    "The ", "United States", "New York", "Los Angeles", "San Francisco",
    "North ", "South ", "West ", "East ", "Great ", "White House",
    "King's Cross", "Area 51", "Baker Street", "Yellow River",
)


def find_name_candidates(text):
    if not text:
        return []
    matches = NAME_SUBSTRING_PATTERN.findall(text)
    return [
        m for m in matches
        if not any(m.startswith(s) for s in SKIP_IF_STARTS_WITH)
        and not any(ch.isdigit() for ch in m)
    ]


def best_knowledge_sentence(correct_answers_field):
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

    best_answer_names = find_name_candidates(best_answer)
    if not best_answer_names:
        continue

    # Find an incorrect answer that ALSO contains a name-shaped
    # substring, ideally a DIFFERENT one from the correct answer's
    matching_incorrect = None
    matching_incorrect_name = None
    for inc in incorrect_answers:
        inc_names = find_name_candidates(inc)
        for n in inc_names:
            if n not in best_answer_names:
                matching_incorrect = inc
                matching_incorrect_name = n
                break
        if matching_incorrect:
            break

    if not matching_incorrect:
        continue

    knowledge = best_knowledge_sentence(row["Correct Answers"])
    if not knowledge or len(knowledge) < 20:
        continue

    candidates.append({
        "question": row["Question"],
        "best_answer": best_answer,
        "best_answer_name_found": best_answer_names[0],
        "incorrect_answer": matching_incorrect,
        "incorrect_answer_name_found": matching_incorrect_name,
        "knowledge": knowledge,
        "category": row["Category"],
    })

print(f"Candidates found (name-shaped substring in both fields): {len(candidates)}")

with open(OUTPUT_PATH, "w") as f:
    for i, c in enumerate(candidates):
        block = (
            f"{'='*70}\n"
            f"CANDIDATE #{i}  (Category: {c['category']})\n"
            f"{'='*70}\n"
            f"Question: {c['question']}\n"
            f"True Answer (full): {c['best_answer']}\n"
            f"  -> name-like substring detected: {c['best_answer_name_found']}\n"
            f"Incorrect Answer (full): {c['incorrect_answer']}\n"
            f"  -> name-like substring detected: {c['incorrect_answer_name_found']}\n"
            f"Knowledge (from Correct Answers field): {c['knowledge']}\n\n"
        )
        f.write(block)

print(f"Saved candidate list to: {OUTPUT_PATH}")
print("This list will include some false positives (place names, "
      "titles) alongside real person candidates — review and pick "
      "the clean ones, same as before.")
