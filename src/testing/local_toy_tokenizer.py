from __future__ import annotations

from dataclasses import dataclass

from src.core.constants import RESERVED_TOKENS


@dataclass
class LocalToyTokenizer:
    token_to_id: dict[str, int]

    @classmethod
    def build_from_texts(cls, texts: list[str]) -> "LocalToyTokenizer":
        vocab = sorted({token for text in texts for token in text.split()})
        token_to_id = {token: idx for idx, token in enumerate(RESERVED_TOKENS)}
        next_index = len(token_to_id)
        for token in vocab:
            token_to_id[token] = next_index
            next_index += 1
        return cls(token_to_id=token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.unk_token_id) for token in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return " ".join(id_to_token.get(token_id, "<unk>") for token_id in token_ids)

