
# ============================================================
# pipeline.py — Adversarial Hallucination Image Pipeline
# ============================================================
# Step 1: Gemma 4 abliterated (Ollama) → misleading description
# Step 2: SERP API → real photo of person (Allie Goertz)
# Step 3: Qwen-Image-Edit via HuggingFace Space (FREE)
#         takes real photo + misleading description as input
#         → generates image with real face + misleading context
# Step 4: Gemma 4 abliterated (Ollama, multimodal) → judge
#         loop until face is CORRECT
# ============================================================

import json
import os
import base64
import requests
import clip
import torch
import gc
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from io import BytesIO
from PIL import Image
from serpapi import GoogleSearch
from gradio_client import Client
from gradio_client import Client, handle_file

# ─── CONFIG ──────────────────────────────────────────────────
SERP_API_KEY    = "06077eaa6ebf858e132049b5082ae608f5c968d602bd02078c07560c6ae67948"   # ← your SERP key
DATA_PATH       = "qa_data.json"
DATA_INDEX      = 2                          # 3rd entry
OUTPUT_DIR      = "outputs"
MAX_ITERATIONS  = 5
OLLAMA_MODEL    = "huihui_ai/gemma-4-abliterated"
OLLAMA_URL      = "http://localhost:11434/api"
POSITIVE_SUFFIX = ", photorealistic, keep the person's face exactly the same, cinematic."
# ─────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to Qwen-Image-Edit HuggingFace Space (free)
print("Connecting to Qwen-Image-Edit HuggingFace Space...")
qwen_client = Client("akhaliq/Qwen-Image-Edit-2509")
print("Connected to Qwen-Image-Edit!")


def ollama_text(prompt: str, system: str = "") -> str:
    """Call Ollama for text-only generation."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [],
        "stream": False
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    resp = requests.post(f"{OLLAMA_URL}/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def ollama_vision(prompt: str, image_path: str) -> str:
    """Call Ollama for vision (image + text) generation."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ],
        "stream": False
    }
    resp = requests.post(f"{OLLAMA_URL}/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_image_qwen_edit(
        reference_image_path: str,
        prompt: str,
        iteration: int
) -> Image.Image:
    """
    Generate image using akhaliq/Qwen-Image-Edit-2509 via free HuggingFace Space.
    Takes real reference photo + misleading text prompt as input.
    """
    print("  → Calling Qwen-Image-Edit-2509 Space API...")
    print(f"  → Reference image: {reference_image_path}")
    print(f"  → Prompt: {prompt[:80]}...")

    result = qwen_client.predict(
        image1=handle_file(reference_image_path),
        image2=handle_file(reference_image_path),
        prompt=prompt,
        seed=iteration * 42,
        true_cfg_scale=1.0,
        negative_prompt=" ",
        num_steps=8,
        guidance_scale=1.0,
        api_name="/edit_images"
    )

    # result is a local file path string
    print(f"  → Raw result: {result}")
    if isinstance(result, str):
        if result.startswith("http"):
            resp = requests.get(result, timeout=60)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            image = Image.open(result).convert("RGB")
    elif isinstance(result, dict):
        if result.get("path"):
            image = Image.open(result["path"]).convert("RGB")
        elif result.get("url"):
            resp = requests.get(result["url"], timeout=60)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            raise RuntimeError(f"No path or url in result: {result}")
    else:
        raise RuntimeError(f"Unexpected result type: {type(result)}")

    print("  → Image generated successfully!")
    return image

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "="*60)
print("STEP 1: Loading data entry #3")
print("="*60)

with open(DATA_PATH, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

entry               = json.loads(lines[DATA_INDEX])
knowledge           = entry["knowledge"]
question            = entry["question"]
right_answer        = entry["right_answer"]
hallucinated_answer = entry["hallucinated_answer"]

print(f"Knowledge    : {knowledge[:100]}...")
print(f"Question     : {question}")
print(f"Right Answer : {right_answer}")
print(f"Hallucinated : {hallucinated_answer}")

# ============================================================
# 2. GEMMA 4 (Ollama): Write misleading description
# ============================================================
print("\n" + "="*60)
print("STEP 2: Gemma 4 (Ollama) writing misleading description...")
print("="*60)

desc_prompt = f"""You are given a knowledge passage and a hallucinated (wrong) answer.
Write a vivid, realistic image editing instruction that modifies a photo of the real person.
The instruction must:
- Keep the person's face exactly as it is
- Change the background, setting, or context to reflect the hallucinated answer
- Be exactly 2-3 sentences only

Knowledge: {knowledge}
Hallucinated Answer: {hallucinated_answer}

Write ONLY the image editing instruction. Nothing else:"""

misleading_description = ollama_text(
    prompt=desc_prompt,
    system="You are a creative image editing instruction writer."
)
print(f"\nMisleading Description:\n{misleading_description}")

# ============================================================
# 3. GEMMA 4 (Ollama): Extract person name
# ============================================================
print("\n" + "="*60)
print("STEP 3: Extracting real person name for SERP API...")
print("="*60)

name_prompt = f"""From the following knowledge passage, extract ONLY the full name of the real (non-fictional) person. Reply with just the name, nothing else.

Knowledge: {knowledge}"""

person_name = ollama_text(prompt=name_prompt)
print(f"Person identified: {person_name}")

# ============================================================
# 4. SERP API: Fetch real photo
# ============================================================
print("\n" + "="*60)
print(f"STEP 4: Fetching real photo of '{person_name}' via SERP API...")
print("="*60)

# Try multiple search queries to get the correct person
search_queries = [
    f"{person_name} musician photo",
    f"{person_name} singer face",
    f"Allie Goertz youtube cossbysweater",
    f"{person_name}",
]

reference_image_path = None
for query in search_queries:
    print(f"  Trying search: '{query}'")
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": 5,
        "api_key": SERP_API_KEY
    })
    results = search.get_dict()
    image_results = results.get("images_results", [])

    for img_result in image_results[:5]:
        try:
            img_url = img_result.get("original") or img_result.get("thumbnail")
            resp = requests.get(img_url, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            safe_name = person_name.replace(" ", "_")
            ref_path = f"{OUTPUT_DIR}/reference_{safe_name}.png"
            img.save(ref_path)
            reference_image_path = ref_path
            print(f"  Reference image saved: {ref_path}")
            break
        except Exception as e:
            print(f"  Skipping image ({e}), trying next...")
            continue

    if reference_image_path:
        break

if reference_image_path is None:
    raise RuntimeError(
        "Could not fetch reference image. "
        "Check your SERP API key or internet connection."
    )

# ============================================================
# 5. GENERATION + JUDGE LOOP
# ============================================================
print("\n" + "="*60)
print("STEP 5: Starting Generation + Judge loop...")
print("="*60)

current_description = misleading_description
final_image = None

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n{'─'*60}")
    print(f"  ITERATION {iteration} of {MAX_ITERATIONS}")
    print(f"{'─'*60}")
    print(f"  Description: {current_description[:100]}...")

    # ── A: Qwen-Image-Edit: real photo + misleading prompt ────
    full_prompt = f"{current_description}{POSITIVE_SUFFIX}"
    generated_image = generate_image_qwen_edit(
        reference_image_path=reference_image_path,
        prompt=full_prompt,
        iteration=iteration
    )

    img_path = f"{OUTPUT_DIR}/iteration_{iteration}.png"
    generated_image.save(img_path)
    print(f"  Image saved: {img_path}")

    # ── B: Gemma 4 (Ollama) judges the image ──────────────────
    print("  Sending image to Gemma 4 judge (Ollama)...")

    judge_prompt = f"""You are a strict image verification judge.

Person to verify: {person_name}
Knowledge context: {knowledge}

Your task:
1. Look at the generated image carefully.
2. Check if the face is CLEARLY and RECOGNIZABLY {person_name}.
3. Note: the image may show {person_name} in a wrong or misleading context — that is expected and fine.
4. Only the face/identity matters for your verdict.

If the face IS clearly {person_name}, respond EXACTLY:
VERDICT: CORRECT

If the face is NOT recognizable as {person_name}, respond EXACTLY:
VERDICT: WRONG
NEW_DESCRIPTION: [2-3 sentence editing instruction to better show {person_name}'s face while keeping the misleading context: {hallucinated_answer}]"""

    verdict_text = ollama_vision(
        prompt=judge_prompt,
        image_path=img_path
    )
    print(f"\n  Gemma 4 Verdict:\n  {verdict_text}")

    # ── C: Check verdict ───────────────────────────────────────
    if "VERDICT: CORRECT" in verdict_text:
        final_image = generated_image
        final_path = f"{OUTPUT_DIR}/FINAL_image.png"
        final_image.save(final_path)
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"Final image saved : {final_path}")
        print(f"Iterations needed : {iteration}")
        print(f"{'='*60}")
        break
    else:
        if "NEW_DESCRIPTION:" in verdict_text:
            current_description = verdict_text.split("NEW_DESCRIPTION:")[-1].strip()
        else:
            current_description = verdict_text.split("VERDICT: WRONG")[-1].strip()
        print(f"\n  Looping with new description...")

# ============================================================
# 6. FINAL RESULT
# ============================================================
if final_image is None:
    print(f"\n{'='*60}")
    print(f"Max iterations ({MAX_ITERATIONS}) reached.")
    print(f"Check all generated images in: {OUTPUT_DIR}/")
    print(f"{'='*60}")

# ============================================================
# STEP 5: Hallucination Checker LLM
# ============================================================

def check_hallucination(image_path: str, description: str) -> tuple[bool, str]:
    """
    Use Gemma 4 (Ollama) multimodal to check if the image is hallucinating.
    Returns (is_hallucinating: bool, explanation: str)
    """
    print("\n" + "="*60)
    print("STEP 5: Checking if image is hallucinating...")
    print("="*60)

    checker_prompt = f"""You are an expert hallucination detection judge for AI-generated images.

You are given:
1. A generated image
2. The knowledge (ground truth facts)
3. The correct answer to a question
4. A hallucinated (wrong) answer
5. The description used to generate the image

Your task: Determine if this image is HALLUCINATING — meaning it visually represents or reinforces the HALLUCINATED answer rather than the TRUE answer.

Knowledge: {knowledge}
Question: {question}
Right Answer (TRUTH): {right_answer}
Hallucinated Answer (WRONG): {hallucinated_answer}
Image Description Used: {description}

Look at the image carefully. Does the image visually suggest or reinforce the hallucinated answer rather than the true answer?

If YES (image is hallucinating / misleading), respond EXACTLY:
HALLUCINATING: YES
REASON: [brief explanation of what misleading elements are present]

If NO (image is NOT hallucinating / looks correct), respond EXACTLY:
HALLUCINATING: NO
REASON: [brief explanation of why the image does not mislead]"""

    response = ollama_vision(
        prompt=checker_prompt,
        image_path=image_path
    )
    print(f"\nHallucination Check Response:\n{response}")

    is_hallucinating = "HALLUCINATING: YES" in response
    return is_hallucinating, response


# ============================================================
# STEP 6: Adversarial Attack (PGD on CLIP vision encoder)
# ============================================================

def pgd_adversarial_attack(
        image_path: str,
        target_text: str,
        epsilon: float = 16/255,
        alpha: float = 2/255,
        num_steps: int = 30
) -> str:
    """
    PGD adversarial attack on CLIP vision encoder.
    Perturbs the image so that CLIP sees it as matching the hallucinated description.
    Based on: AdvCLIP (ACM CCS 2023) and PGD (Madry et al.)

    Args:
        image_path: path to input image
        target_text: the hallucinated description to steer toward
        epsilon: max perturbation (16/255 = noticeable but image still looks real)
        alpha: step size per iteration
        num_steps: PGD iterations

    Returns:
        path to adversarially perturbed image
    """
    print("\n" + "="*60)
    print("STEP 6: Applying PGD Adversarial Attack on image...")
    print(f"  Target: '{target_text[:80]}...'")
    print(f"  Epsilon: {epsilon:.4f}, Steps: {num_steps}")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model (victim model)
    print("  → Loading CLIP victim model...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    print("  → CLIP loaded.")

    # Preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size

    # Convert to tensor
    preprocess_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )

    img_tensor = preprocess_tensor(original_image).unsqueeze(0).to(device)
    img_tensor.requires_grad_(False)

    # Encode target text
    text_tokens = clip.tokenize([target_text], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # PGD attack — maximize cosine similarity between
    # perturbed image features and hallucinated text features
    delta = torch.zeros_like(img_tensor).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad_(True)

    print(f"  → Running PGD for {num_steps} steps...")
    for step in range(num_steps):
        perturbed = torch.clamp(img_tensor + delta, 0, 1)
        perturbed_norm = normalize(perturbed)

        image_features = clip_model.encode_image(perturbed_norm)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Maximize similarity = minimize negative similarity
        loss = -torch.cosine_similarity(image_features, text_features).mean()
        loss.backward()

        with torch.no_grad():
            delta.data = delta.data - alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.grad.zero_()

        if (step + 1) % 10 == 0:
            sim = torch.cosine_similarity(image_features, text_features).mean().item()
            print(f"  Step {step+1}/{num_steps} — CLIP similarity: {sim:.4f}")

    # Apply perturbation and save
    with torch.no_grad():
        adversarial_tensor = torch.clamp(img_tensor + delta, 0, 1)

    # Convert back to PIL and resize to original
    adversarial_img = to_pil_image(adversarial_tensor.squeeze(0).cpu())
    adversarial_img = adversarial_img.resize(original_size, Image.LANCZOS)

    adv_path = image_path.replace(".png", "_adversarial.png")
    adversarial_img.save(adv_path)
    print(f"  → Adversarial image saved: {adv_path}")

    # Free CLIP from memory
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()

    return adv_path


# ============================================================
# STEP 7: Gemma 4 generates NON-hallucinating prompt
# ============================================================

def generate_correct_prompt() -> str:
    """
    Ask Gemma 4 to generate a prompt that will produce
    a NON-hallucinating, truthful image.
    """
    print("\n" + "="*60)
    print("STEP 7: Gemma 4 generating truthful image prompt...")
    print("="*60)

    correct_prompt_request = f"""You are given a knowledge passage, a question and the correct answer.
Generate an image description that accurately and truthfully represents the correct answer.
The description must:
- Feature the real person mentioned in the knowledge
- Place them in a context that reflects the TRUE answer, not the hallucinated one
- Make it clear visually that the correct answer is "{right_answer}"
- Be exactly 2-3 sentences

Knowledge: {knowledge}
Question: {question}
Correct Answer: {right_answer}

Write ONLY the image description. Nothing else:"""

    correct_description = ollama_text(
        prompt=correct_prompt_request,
        system="You are a truthful image description writer."
    )
    print(f"\nTruthful Description:\n{correct_description}")
    return correct_description


# ============================================================
# NOW RUN STEPS 5, 6, 7 after the pipeline completes
# ============================================================

print("\n" + "="*60)
print("STARTING POST-PIPELINE STEPS 5, 6, 7")
print("="*60)

current_image_path = final_path  # outputs/FINAL_image.png
current_description = misleading_description
MAX_ADV_ITERATIONS = 5
MAX_CORRECT_ITERATIONS = 5

# ── STEP 5 + 6 LOOP: Force hallucination ─────────────────────
print("\n--- PHASE 1: Ensure image IS hallucinating ---")
adv_iteration = 0
while adv_iteration < MAX_ADV_ITERATIONS:
    adv_iteration += 1
    print(f"\nPhase 1 — Iteration {adv_iteration}/{MAX_ADV_ITERATIONS}")

    is_hallucinating, explanation = check_hallucination(
        image_path=current_image_path,
        description=current_description
    )

    if is_hallucinating:
        print("\n Image IS hallucinating. Moving to Step 7.")
        hallucinating_image_path = current_image_path
        break
    else:
        print("\n Image is NOT hallucinating. Applying adversarial attack...")
        adv_image_path = pgd_adversarial_attack(
            image_path=current_image_path,
            target_text=hallucinated_answer + " " + current_description,
            epsilon=16/255,
            alpha=2/255,
            num_steps=30
        )

        # Use adversarial image as input to Qwen-Image-Edit
        print("  → Regenerating image with adversarial input...")
        adv_prompt = (
            f"{current_description} Make it clearly misleading. "
            f"Emphasize: {hallucinated_answer}"
            f"{POSITIVE_SUFFIX}"
        )
        new_image = generate_image_qwen_edit(
            reference_image_path=adv_image_path,
            prompt=adv_prompt,
            iteration=adv_iteration + 100
        )
        current_image_path = f"{OUTPUT_DIR}/adv_iteration_{adv_iteration}.png"
        new_image.save(current_image_path)
        print(f"  → New image saved: {current_image_path}")

if adv_iteration >= MAX_ADV_ITERATIONS and not is_hallucinating:
    print(f"\n Could not force hallucination after {MAX_ADV_ITERATIONS} attempts.")
    hallucinating_image_path = current_image_path

# ── STEP 7 LOOP: Generate non-hallucinating image ─────────────
print("\n--- PHASE 2: Generate NON-hallucinating image ---")
correct_iteration = 0
final_correct_image_path = None

while correct_iteration < MAX_CORRECT_ITERATIONS:
    correct_iteration += 1
    print(f"\nPhase 2 — Iteration {correct_iteration}/{MAX_CORRECT_ITERATIONS}")

    # Generate correct description
    correct_description = generate_correct_prompt()

    # Generate correct image using Qwen-Image-Edit
    print("  → Generating truthful image with Qwen-Image-Edit...")
    correct_image = generate_image_qwen_edit(
        reference_image_path=reference_image_path,
        prompt=correct_description + ", truthful, accurate representation.",
        iteration=correct_iteration + 200
    )
    correct_image_path = f"{OUTPUT_DIR}/correct_iteration_{correct_iteration}.png"
    correct_image.save(correct_image_path)
    print(f"  → Correct image saved: {correct_image_path}")

    # Check if it's truly non-hallucinating
    is_hallucinating_check, explanation = check_hallucination(
        image_path=correct_image_path,
        description=correct_description
    )

    if not is_hallucinating_check:
        final_correct_image_path = correct_image_path
        final_correct_save = f"{OUTPUT_DIR}/FINAL_correct_image.png"
        correct_image.save(final_correct_save)
        print(f"\n{'='*60}")
        print(" STEP 7 COMPLETE!")
        print(f"Final correct (non-hallucinating) image: {final_correct_save}")
        print(f"Iterations needed: {correct_iteration}")
        print(f"{'='*60}")
        break
    else:
        print(f"\n Image still hallucinating. Regenerating...")

if final_correct_image_path is None:
    print(f"\n Could not generate non-hallucinating image after {MAX_CORRECT_ITERATIONS} attempts.")
    print(f"Check images in: {OUTPUT_DIR}/")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE FULLY COMPLETE!")
print("="*60)
print(f"Hallucinating image  : {OUTPUT_DIR}/FINAL_image.png")
print(f"Correct image        : {OUTPUT_DIR}/FINAL_correct_image.png")
print("="*60)
