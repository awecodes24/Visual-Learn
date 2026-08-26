from __future__ import annotations
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are VizoLearn, a careful visual AI tutor. Explain things clearly and educationally.
Only make visual claims supported by the object class and visual description supplied by the system.
Never invent a brand, exact model, material, internal component, location, or function from appearance alone.
Prefer concise but useful explanations. Structure initial explanations with:
1. What is it?
2. What is it used for?
3. How does it work?
4. Why is it useful?
5. Interesting fact
For uncertainty, explicitly say that the detail cannot be determined from the image alone.
"""


@lru_cache(maxsize=1)
def _load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    return tokenizer, model


def _generate(messages: list[dict[str, str]], max_new_tokens: int = 320) -> str:
    tokenizer, model = _load_qwen()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        # Greedy decoding (do_sample=False) is deliberate: it gives repeatable,
        # non-random explanations for the same object. `temperature` only
        # affects sampling-based generation, so it is intentionally omitted
        # here rather than paired with do_sample=False.
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def explain_object(object_name: str, visual_context: str) -> str:
    user = f"Selected object class: {object_name}\nVisual description: {visual_context}\nExplain this object for a student."
    return _generate([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])


def answer_question(object_name: str, visual_context: str, history: list[dict[str, str]], question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": f"Current object: {object_name}. Visual context: {visual_context}. Student question: {question}"})
    return _generate(messages, max_new_tokens=220)