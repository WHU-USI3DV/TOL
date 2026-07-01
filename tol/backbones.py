from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from CLIP import clip


CLIP_MODEL_NAMES = {
    "clip-b16": "ViT-B/16",
    "clip-b32": "ViT-B/32",
    "vit-b-16": "ViT-B/16",
    "vit-b-32": "ViT-B/32",
}


def _is_clip_model(name: str) -> bool:
    return name.lower() in CLIP_MODEL_NAMES or name.lower().startswith("clip")


def build_backbone(name: str, device: str = "cuda", jit: bool = False):
    key = name.lower()
    if key in CLIP_MODEL_NAMES:
        model, _ = clip.load(CLIP_MODEL_NAMES[key], device=device, jit=jit)
        return model
    return AutoModel.from_pretrained(name).to(device)


def build_preprocess(name: str, device: str = "cuda", context_length: int = 77):
    key = name.lower()
    if key in CLIP_MODEL_NAMES:
        _, preprocess = clip.load(CLIP_MODEL_NAMES[key], device=device, jit=False)

        def clip_preprocess(image, text):
            image_tensor = preprocess(image)
            text_tensor = clip.tokenize(text, context_length=context_length, truncate=True).squeeze(0)
            return image_tensor, text_tensor

        return clip_preprocess

    processor = AutoProcessor.from_pretrained(name)

    def hf_preprocess(image, text):
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return inputs["pixel_values"].squeeze(0), inputs["input_ids"].squeeze(0)

    return hf_preprocess


def infer_backbone_dims(model):
    if hasattr(model, "visual"):
        image_width = getattr(model.visual, "width", None)
        if image_width is None and hasattr(model.visual, "conv1"):
            image_width = model.visual.conv1.out_channels
        if image_width is None and hasattr(model.visual, "transformer"):
            image_width = getattr(model.visual.transformer, "width", None)
        text_width = getattr(model, "text_projection", None)
        text_dim = text_width.shape[1] if text_width is not None else image_width
        if image_width is None or text_dim is None:
            raise RuntimeError("Cannot infer CLIP backbone dimensions from the loaded model.")
        return image_width, text_dim
    vision_cfg = model.vision_model.config
    text_cfg = model.text_model.config
    return vision_cfg.hidden_size, text_cfg.hidden_size


def _rename_legacy_checkpoint_key(key: str):
    if key.startswith("module."):
        key = key[len("module.") :]

    if key.startswith("model.text_fusion.") or key.startswith("text_fusion."):
        return None

    if key.startswith("model.base_model."):
        return "encoder.base_model." + key[len("model.base_model.") :]
    if key.startswith("base_model."):
        return "encoder.base_model." + key[len("base_model.") :]

    conv_prefixes = {
        "model.conv.conv.": "encoder.conv.net.0.",
        "model.conv.bn.": "encoder.conv.net.1.",
        "conv.conv.": "encoder.conv.net.0.",
        "conv.bn.": "encoder.conv.net.1.",
    }
    for old_prefix, new_prefix in conv_prefixes.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]

    if key.startswith("att.norm_txt."):
        return "registration.norm_text." + key[len("att.norm_txt.") :]
    if key.startswith("att."):
        return "registration." + key[len("att.") :]
    if key.startswith("fc."):
        return "offset_head." + key[len("fc.") :]

    return key


def convert_legacy_state_dict(state):
    converted = {}
    for key, value in state.items():
        new_key = _rename_legacy_checkpoint_key(key)
        if new_key is not None:
            converted[new_key] = value
    return converted


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, strict: bool = True):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = convert_legacy_state_dict(state)
    model_state = model.state_dict()
    for key, value in list(state.items()):
        if key in model_state and value.shape != model_state[key].shape and value.numel() == model_state[key].numel():
            state[key] = value.reshape(model_state[key].shape)
    model.load_state_dict(state, strict=strict)
    return model
