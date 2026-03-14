from __future__ import annotations

import importlib
import sys
import types

import torch

from tests.helpers import REPO_ROOT, load_yaml


def test_load_documents_supports_remote_dataset_via_fake_datasets_module(monkeypatch) -> None:
    calls: list[tuple[tuple, dict]] = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {"text": "alpha\x00 beta"},
            {"content": "gamma\r\ndelta"},
            {"raw_content": "epsilon"},
            {"text": "   "},
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    import src.data.dataset_stream as dataset_stream

    importlib.reload(dataset_stream)
    config = load_yaml(REPO_ROOT / "configs" / "base.yaml")
    config["data"]["local_path"] = None
    config["data"]["dataset_name"] = "fake/remote-corpus"
    config["data"]["dataset_config"] = "stub"

    documents = dataset_stream.load_documents(config)

    assert documents == ["alpha beta", "gamma\ndelta", "epsilon"]
    assert calls, "expected the remote path to use datasets.load_dataset"
    assert calls[0][0][0] == "fake/remote-corpus"


def test_backbone_wrapper_supports_remote_transformers_path_via_fake_module(monkeypatch) -> None:
    class FakeTokenizer:
        vocab_size = 11

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 4)
            self.config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=4, vocab_size=11)

        def to(self, device):
            self._device = device
            return self

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            batch, seq_len = input_ids.shape
            hidden = self.embedding(input_ids)
            hidden_states = [
                hidden * 0.0,
                hidden + 1.0,
                hidden + 2.0,
            ]
            logits = torch.randn(batch, seq_len, 11, device=input_ids.device)
            return types.SimpleNamespace(hidden_states=hidden_states, logits=logits)

        def get_output_embeddings(self):
            return types.SimpleNamespace(weight=self.embedding.weight)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )

    import src.models.backbone_wrapper as backbone_wrapper

    importlib.reload(backbone_wrapper)
    wrapper = backbone_wrapper.BackboneWrapper("EleutherAI/pythia-410m", precision="bf16", device="cpu")
    outputs = wrapper.forward_hidden(torch.tensor([[1, 2, 3]]), return_base_logits=True)

    assert wrapper.model_slug() == "EleutherAI-pythia-410m"
    assert len(outputs.hidden_states) == 2, "embedding output should be excluded from transformer-block hidden states"
    assert tuple(outputs.hidden_states[0].shape) == (1, 3, 4)
    assert tuple(outputs.base_logits.shape) == (1, 3, 11)
    assert wrapper.vocab_size() == 11
    assert all(not parameter.requires_grad for parameter in wrapper.model.parameters())


def test_backbone_wrapper_uses_torch_dtype_for_cuda_remote_load(monkeypatch) -> None:
    model_calls: list[tuple[tuple, dict]] = []

    class FakeTokenizer:
        vocab_size = 11

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 4)
            self.config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=4, vocab_size=11)

        def get_output_embeddings(self):
            return types.SimpleNamespace(weight=self.embedding.weight)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model_calls.append((args, kwargs))
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    import src.models.backbone_wrapper as backbone_wrapper

    importlib.reload(backbone_wrapper)
    backbone_wrapper.BackboneWrapper("EleutherAI/pythia-410m", precision="bf16", device="cuda")

    assert model_calls, "expected the remote model loader to be called"
    _, kwargs = model_calls[0]
    assert kwargs["torch_dtype"] == torch.bfloat16
    assert kwargs["device_map"] == {"": "cuda:0"}
    assert "dtype" not in kwargs


def test_backbone_wrapper_vocab_size_uses_output_embedding_rows_for_remote_models(monkeypatch) -> None:
    class FakeTokenizer:
        vocab_size = 11

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 4)
            self.config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=4, vocab_size=13)

        def to(self, device):
            self._device = device
            return self

        def get_output_embeddings(self):
            return types.SimpleNamespace(weight=self.embedding.weight)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )

    import src.models.backbone_wrapper as backbone_wrapper

    importlib.reload(backbone_wrapper)
    wrapper = backbone_wrapper.BackboneWrapper("EleutherAI/pythia-410m", precision="bf16", device="cpu")

    assert wrapper.vocab_size() == 13
