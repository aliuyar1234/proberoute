from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import torch
from torch import nn

from src.core.constants import SMOKE_BACKBONE_NAME
from src.testing.local_toy_backbone import LocalToyBackbone
from src.testing.local_toy_tokenizer import LocalToyTokenizer


@dataclass
class BackboneOutputs:
    hidden_states: list[torch.Tensor]
    base_logits: torch.Tensor | None
    attention_mask: torch.Tensor | None


class BackboneWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        precision: str,
        device: str,
        *,
        tokenizer=None,
        smoke_mode: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.precision = precision
        self.device_name = device
        self.smoke_mode = smoke_mode or model_name == SMOKE_BACKBONE_NAME
        self._uses_device_map = False
        if self.smoke_mode:
            if tokenizer is None:
                raise ValueError("The local toy backbone requires a LocalToyTokenizer")
            self._tokenizer = tokenizer
            self.model = LocalToyBackbone(vocab_size=tokenizer.vocab_size)
        else:
            self._tokenizer = tokenizer or self._load_remote_tokenizer(model_name)
            self.model = self._load_remote_model(model_name)
        if self.smoke_mode or not self._uses_device_map:
            self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _precision_dtype(self):
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self.precision, torch.float32)

    def _load_remote_model(self, model_name: str):
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError("Remote backbone loading requires the `transformers` package to be installed.") from exc
        kwargs = {
            "output_hidden_states": True,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            kwargs["torch_dtype"] = self._precision_dtype()
            kwargs["device_map"] = {"": f"cuda:{self.device.index or 0}"}
            self._uses_device_map = True
        return transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _load_remote_tokenizer(self, model_name: str):
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError("Remote tokenizer loading requires the `transformers` package to be installed.") from exc
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def device(self) -> torch.device:
        if self.device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def tokenizer(self):
        return self._tokenizer

    def num_layers(self) -> int:
        if self.smoke_mode:
            return int(self.model.num_layers)
        return int(getattr(self.model.config, "num_hidden_layers"))

    def hidden_size(self) -> int:
        if self.smoke_mode:
            return int(self.model.hidden_size)
        return int(getattr(self.model.config, "hidden_size"))

    def vocab_size(self) -> int:
        if self.smoke_mode:
            return int(self._tokenizer.vocab_size)
        return int(self.unembedding_weight().shape[0])

    def model_slug(self) -> str:
        return "local-toy-gpt" if self.smoke_mode else self.model_name.replace("/", "-")

    def unembedding_weight(self) -> torch.Tensor:
        if self.smoke_mode:
            return self.model.unembed_weight
        output_embedding = self.model.get_output_embeddings()
        if output_embedding is None:
            raise ValueError("Remote model does not expose output embeddings")
        return output_embedding.weight

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_base_logits: bool = False,
    ) -> BackboneOutputs:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        with torch.no_grad():
            if self.smoke_mode:
                hidden_states, base_logits = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_base_logits=return_base_logits,
                )
            else:
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "output_hidden_states": True,
                }
                try:
                    outputs = self.model(return_dict=True, **forward_kwargs)
                except TypeError:
                    outputs = self.model(**forward_kwargs)
                all_hidden_states = list(outputs.hidden_states or [])
                hidden_states = all_hidden_states[1:]
                base_logits = outputs.logits if return_base_logits else None
        return BackboneOutputs(hidden_states=hidden_states, base_logits=base_logits, attention_mask=attention_mask)
