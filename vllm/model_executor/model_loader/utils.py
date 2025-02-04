"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry
import os


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    support_nn_architectures = ['LlamaForCausalLM', 'QWenLMHeadModel', 'Qwen2ForCausalLM', 'ChatGLMModel', 'BaichuanForCausalLM']  
    if any(arch in architectures for arch in support_nn_architectures): 
        if os.getenv('LLAMA_NN') != '0': 
            os.environ['LLAMA_NN'] = '1'
        if os.getenv('GEMM_PAD') != '1': 
            os.environ['GEMM_PAD'] = '0'
        if os.getenv('FA_PAD') != '1': 
            os.environ['FA_PAD'] = '0'
    else:
        os.environ['LLAMA_NN'] = '0'
        os.environ['GEMM_PAD'] = '0'
        os.environ['FA_PAD'] = '0'
        
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    if (model_config.quantization is not None
            and model_config.quantization != "fp8"
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]
