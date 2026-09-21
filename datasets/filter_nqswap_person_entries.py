# ============================================================
# filter_nqswap_person_entries.py
# ============================================================
# Scans the FULL downloaded NQ-Swap-dev.json for entries where
# BOTH the true answer (org_answer) and the substituted/false
# answer (sub_answer) look like real person names — the shape this
# pipeline needs for its photo-based visual attack.
#
# Heuristic: an answer is a "likely person name" if it's 2-4 words,
# every word starts with a capital letter, and it contains no
# digits. This isn't perfect (will miss single-name mononyms, will
# occasionally include non-person proper nouns like band names or
# multi-word places), so the output is a CANDIDATE list for manual
# review, not a final selection.
#
# Usage:
#   python3 filter_nqswap_person_entries.py
# ============================================================

import json
import re

INPUT_PATH = "NQ-Swap-dev.json"
OUTPUT_PATH = "nqswap_person_candidates.txt"

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
    if word_count < 2 or word_count > 4:
        return False
    return True


candidates = []

with open(INPUT_PATH, "r") as f:
    for line in f:
        row = json.loads(line)
        org_answers = row.get("org_answer", [])
        sub_answers = row.get("sub_answer", [])

        if not org_answers or not sub_answers:
            continue

        org_answer = org_answers[0]
        sub_answer = sub_answers[0]

        if looks_like_person_name(org_answer) and looks_like_person_name(sub_answer):
            candidates.append({
                "question": row["question"],
                "org_answer": org_answer,
                "sub_answer": sub_answer,
                "org_context": row["org_context"],
                "sub_context": row["sub_context"],
            })

print(f"Candidates found (both answers look like person names): {len(candidates)}")
print()

# Deduplicate by question (many rows share the same question with
# different substitutions — keep only the first occurrence)
seen_questions = set()
deduped = []
for c in candidates:
    if c["question"] not in seen_questions:
        seen_questions.add(c["question"])
        deduped.append(c)

print(f"After deduplicating by question: {len(deduped)} unique candidates")
print()

with open(OUTPUT_PATH, "w") as f:
    for i, c in enumerate(deduped):
        block = (
            f"{'='*70}\n"
            f"CANDIDATE #{i}\n"
            f"{'='*70}\n"
            f"Question: {c['question']}\n"
            f"True Answer: {c['org_answer']}\n"
            f"Substituted (False) Answer: {c['sub_answer']}\n"
            f"Full Context (true):\n  {c['org_context']}\n\n"
        )
        f.write(block)

print(f"Saved full candidate list to: {OUTPUT_PATH}")
print(f"Open that file, review the {len(deduped)} candidates, and pick")
print(f"whichever ~20-30 look cleanest for the pipeline.")
