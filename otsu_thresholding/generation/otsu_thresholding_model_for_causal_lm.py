import transformers

from common.params import *
from otsu_thresholding.generation.utils import OtsuThresholdingMixin


class OtsuThresholdingModelForCausalLM:
    _SUPPORTED = {
        LLAMA_8B: transformers.LlamaForCausalLM,
        LLAMA_70B: transformers.LlamaForCausalLM,
        MISTRAL_24B: transformers.MistralForCausalLM,
        QWEN_8B: transformers.Qwen3ForCausalLM,
        QWEN_14B: transformers.Qwen3ForCausalLM
    }

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        base_cls = cls._SUPPORTED[model_name]
        otsu_cls = make_otsu_class(base_cls)
        model = otsu_cls.from_pretrained(model_name, **kwargs)
        return model


def make_otsu_class(base_cls):
    class OtsuThresholdingClass(OtsuThresholdingMixin, base_cls):
        pass

    OtsuThresholdingClass.__name__ = f"OtsuThresholdingClassDecoding{base_cls.__name__}"
    return OtsuThresholdingClass
