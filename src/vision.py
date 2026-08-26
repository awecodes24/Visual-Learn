from __future__ import annotations
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def _load_blip():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def describe_image(crop_bgr: np.ndarray) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return "No usable object crop was available."
    import cv2
    image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image)
    processor, model, device = _load_blip()
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    return processor.decode(output[0], skip_special_tokens=True).strip()