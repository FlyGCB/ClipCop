from .base import BaseVLM, build_prompt, parse_yes_no
from .qwen2vl import Qwen2VL
from .internvl2 import InternVL2
from .llava_ov import LLaVAOneVision
from .llama32v import Llama32Vision
from .paligemma2 import PaliGemma2
from .idefics3 import Idefics3
from .deepseek_vl2 import DeepSeekVL2

# Registry: model_name → class
MODEL_REGISTRY = {
    "qwen2vl_7b":    (Qwen2VL,          {"model_id": "Qwen/Qwen2-VL-7B-Instruct"}),
    "internvl2_8b":  (InternVL2,        {"model_id": "OpenGVLab/InternVL2_5-8B"}),
    "llava_ov_7b":   (LLaVAOneVision,   {"model_id": "lmms-lab/llava-onevision-qwen2-7b-ov"}),
    "llama32v_11b":  (Llama32Vision,    {"model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct"}),
    "paligemma2_3b": (PaliGemma2,       {"model_id": "google/paligemma2-3b-pt-448"}),
    "deepseekvl2_small": (DeepSeekVL2, {"model_id": "deepseek-ai/deepseek-vl2-small"}),
}


def get_model(model_name: str, **override_kwargs) -> BaseVLM:
    """
    Instantiate a model by its registry name.

    Args:
        model_name: key in MODEL_REGISTRY
        **override_kwargs: override any default constructor args

    Returns:
        Unloaded model instance (weights loaded lazily on first use)

    Example:
        model = get_model("qwen2vl_7b", load_in_4bit=True)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls, defaults = MODEL_REGISTRY[model_name]
    kwargs = {**defaults, **override_kwargs}
    return cls(model_name=model_name, **kwargs)


__all__ = [
    "BaseVLM", "build_prompt", "parse_yes_no",
    "Qwen2VL", "InternVL2", "LLaVAOneVision",
    "Llama32Vision", "PaliGemma2", "Idefics3",
    "MODEL_REGISTRY", "get_model",
]
