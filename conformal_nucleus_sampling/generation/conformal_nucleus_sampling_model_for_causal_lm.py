import transformers

from common.params import *
from conformal_nucleus_sampling.generation.utils import ConformalNucleusSamplingMixin


class ConformalNucleusSamplingModelForCausalLM:
    _SUPPORTED = {
        LLAMA_8B: transformers.LlamaForCausalLM,
        LLAMA_70B: transformers.LlamaForCausalLM,
        MISTRAL_24B: transformers.MistralForCausalLM,
        QWEN_8B: transformers.Qwen3ForCausalLM,
        QWEN_14B: transformers.Qwen3ForCausalLM
    }

    @classmethod
    def from_pretrained(cls, model_name: str, entropies_quantiles, **kwargs):
        base_cls = cls._SUPPORTED[model_name]
        conformal_cls = make_conformal_class(base_cls)
        model = conformal_cls.from_pretrained(model_name, **kwargs)

        model.entropies_quantiles = entropies_quantiles
        return model


def make_conformal_class(base_cls):
    class ConformalNucleusSamplingClass(ConformalNucleusSamplingMixin, base_cls):
        pass

    ConformalNucleusSamplingClass.__name__ = f"ConformalNucleusSamplingClassDecoding{base_cls.__name__}"
    return ConformalNucleusSamplingClass
