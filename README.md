# Can a Misleading Photo Make an AI Lie About Facts?

This is the code behind our research project, where we tested whether adding a subtle, misleading visual detail to a real photograph can trick a vision-language AI model into giving a wrong answer, even when the correct answer is written right next to it in plain text.

## What we actually did

We took real questions with documented, correct answers (from three separate datasets: HaluEval, NQ-Swap, and TruthfulQA), found a real photo of the person the question was about, and quietly edited that photo to include a small symbolic hint pointing toward a specific wrong answer, a flag, a plaque, a number, nothing that names anyone or spells anything out directly.

Then we showed the AI model two things at once: the edited photo, and a fake newspaper clipping containing the actual true answer. We wanted to know: when a model sees a misleading picture and the correct written fact side by side, which one does it believe?

We tested this on two different AI models (Gemma and Qwen2.5-VL), tried two different ways of defending against it with simple prompt instructions, checked whether tricking one model also fools the other, tested what happens with a much bigger version of each model, and had real people double check our AI judge's decisions to make sure we weren't just trusting one AI to grade another.

## What we found, in short

The models got fooled a lot, sometimes over half the time. Simple prompt-based defenses helped some, but not consistently. Making the model bigger helped a lot more than any prompt trick we tried. And when we compared our subtle photo-editing approach against just writing the wrong answer in plain text on the image, the results depended heavily on which dataset we used, sometimes the obvious fake text worked better, sometimes our subtler version did.

## How this code is organized

There are a lot of files here because this was a long, iterative research project, so here's the general pattern:

- Files starting with `pipeline_strict_batch_` are the actual attack, running the full ten-step process on a set of questions
- Files starting with `pipeline_defense_` or `pipeline_reclassify_` test our two prompt-based defenses
- Files starting with `pipeline_qwen25_` swap in the second AI model to see if the same tricks work on it too
- Files starting with `crosstest_` and `transferability_analysis_` check whether an attack that works on one model also works on the other
- `pipeline_gemma_larger_` and `pipeline_qwen25_larger_` test bigger versions of each model
- `pipeline_baseline_text_overlay` is our comparison test, just writing the wrong answer as plain text instead of a subtle hint
- `mcnemar_test_all_defenses.py` checks whether our defense results are statistically real or just noise
- `build_human_eval_sample.py` and related files set up the human check on our AI judge
- Anything starting with `make_` builds a chart or figure
- `qa_data.json`, `qa_data_nqswap.json`, and `qa_data_truthfulqa.json` are the actual question sets we used

## Running this yourself

You'll need Python 3.12, a way to run the two AI models locally (we used Ollama), and a few common Python packages: `requests`, `Pillow`, and `matplotlib`. Each script is meant to be run on its own, in the order described in our paper's methodology section.

## Citation

If this work is useful to you, please cite our paper:
[citation to be added once the paper is published]

## Questions

Reach out to Zahura Zaman, PhD student in the AI and Security Lab at Boise State University.
