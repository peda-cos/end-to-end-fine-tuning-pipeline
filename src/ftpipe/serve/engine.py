"""Local target and speculative generation engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model.checkpoint import load_checkpoint
from ..model.draft import propose
from ..model.transformer import ModelConfig, forward


class ContentPolicyError(RuntimeError):
    pass


@dataclass
class EngineOutput:
    text: str
    tokens: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    acceptance: float = 0.0
    proposed: int = 0
    accepted: int = 0


@dataclass
class EngineStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    proposed: int = 0
    accepted: int = 0
    waiting: int = 0
    lock: threading.Lock = threading.Lock()


class Engine:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        draft_path: str | Path | None = None,
        guard: Any = None,
        shield: Any = None,
        speculative: bool = True,
        draft_tokens: int = 4,
    ):
        self.params, self.config, self.tokenizer, self.metadata = load_checkpoint(checkpoint)
        self.draft = None
        self.draft_metadata = {}
        if draft_path is not None and Path(draft_path).exists():
            loaded = np.load(Path(draft_path) / "draft.npz", allow_pickle=False)
            self.draft = {name: loaded[name].astype(np.float64, copy=True) for name in loaded.files}
            metadata_path = Path(draft_path) / "metadata.json"
            if metadata_path.exists():
                import json

                self.draft_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.guard = guard
        self.shield = shield
        self.speculative = bool(speculative and self.draft is not None)
        self.draft_tokens = max(1, int(draft_tokens))
        self.stats = EngineStats()
        self.max_batch = 8

    def _prompt_ids(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_bos=True, add_eos=False)

    def check_prompt(self, text: str) -> None:
        if self.guard is not None:
            allowed, label, _ = self.guard.check(text)
            if not allowed:
                raise ContentPolicyError(f"unsafe input: {label}")

    def _shield(self, text: str) -> tuple[str, bool]:
        if self.shield is None:
            return text, False
        allowed, _, _ = self.shield.check(text)
        if not allowed:
            return "I can't help with that request.", True
        return text, False

    def _target_step(self, ids: list[int]) -> np.ndarray:
        tokens = np.asarray([ids], dtype=np.int64)
        cache = forward(self.params, self.config, tokens, positions=np.asarray([[len(ids) - 1]]))
        return cache["logits"][0, 0]

    def generate(self, prompt: str, *, max_new_tokens: int | None = None, temperature: float = 0.0) -> EngineOutput:
        self.check_prompt(prompt)
        ids = self._prompt_ids(prompt)
        max_new = int(max_new_tokens or 12)
        max_ids = self.config.max_seq_len
        prompt_tokens = len(ids)
        proposed = accepted = 0
        finish = "length"
        while len(ids) < max_ids and len(ids) - prompt_tokens < max_new:
            if self.speculative and temperature <= 0:
                cache = forward(self.params, self.config, np.asarray([ids]), positions=np.asarray([[len(ids) - 1]]))
                proposals = propose(
                    self.params,
                    self.config,
                    self.draft,
                    ids,
                    count=min(self.draft_tokens, max_new - (len(ids) - prompt_tokens)),
                    cache=cache,
                    deterministic=True,
                )
                if not proposals:
                    break
                proposed += len(proposals)
                verify_ids = ids + proposals
                verify_ids = verify_ids[:max_ids]
                positions = np.arange(len(ids) - 1, len(ids) - 1 + len(proposals) + 1)
                positions = positions[positions < len(verify_ids)]
                cache = forward(self.params, self.config, np.asarray([verify_ids]), positions=np.asarray([positions]))
                target_tokens = np.argmax(cache["logits"][0], axis=-1).tolist()
                accepted_here = 0
                for index, proposal in enumerate(proposals):
                    if index < len(target_tokens) and target_tokens[index] == proposal:
                        accepted_here += 1
                    else:
                        break
                accepted += accepted_here
                ids.extend(proposals[:accepted_here])
                if accepted_here and ids[-1] == self.tokenizer.eos_id:
                    finish = "stop"
                    break
                if accepted_here < len(target_tokens):
                    ids.append(int(target_tokens[accepted_here]))
                if ids and ids[-1] == self.tokenizer.eos_id:
                    finish = "stop"
                    break
            else:
                logits = self._target_step(ids)
                if temperature <= 0:
                    next_id = int(np.argmax(logits))
                else:
                    scaled = logits / temperature
                    scaled -= scaled.max()
                    probs = np.exp(scaled)
                    probs /= probs.sum()
                    next_id = int(np.random.default_rng().choice(len(probs), p=probs))
                ids.append(next_id)
                if next_id == self.tokenizer.eos_id:
                    finish = "stop"
                    break
        completion = ids[prompt_tokens:]
        text, blocked = self._shield(self.tokenizer.decode(completion))
        if blocked:
            finish = "content_filter"
        with self.stats.lock:
            self.stats.requests += 1
            self.stats.prompt_tokens += prompt_tokens
            self.stats.completion_tokens += len(completion)
            self.stats.proposed += proposed
            self.stats.accepted += accepted
        return EngineOutput(
            text=text,
            tokens=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion),
            finish_reason=finish,
            acceptance=accepted / proposed if proposed else 0.0,
            proposed=proposed,
            accepted=accepted,
        )

    def generate_many(self, prompts: list[str], *, max_new_tokens: int | None = None) -> list[EngineOutput]:
        if not self.speculative:
            return self._generate_target_batch(prompts, max_new_tokens=max_new_tokens or 12)
        return [self.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0) for prompt in prompts]

    def _generate_target_batch(self, prompts: list[str], *, max_new_tokens: int) -> list[EngineOutput]:
        for prompt in prompts:
            self.check_prompt(prompt)
        prompt_ids = [self._prompt_ids(prompt) for prompt in prompts]
        prompt_lengths = [len(ids) for ids in prompt_ids]
        rows = [list(ids) for ids in prompt_ids]
        completion: list[list[int]] = [[] for _ in prompts]
        finished = [False] * len(prompts)
        for _ in range(max_new_tokens):
            width = max(len(row) for row in rows)
            tokens = np.full((len(rows), width), self.tokenizer.pad_id, dtype=np.int64)
            segments = np.full((len(rows), width), -1, dtype=np.int64)
            positions = np.zeros((len(rows), 1), dtype=np.int64)
            for row_index, row in enumerate(rows):
                tokens[row_index, : len(row)] = row
                segments[row_index, : len(row)] = 0
                positions[row_index, 0] = len(row) - 1
            cache = forward(self.params, self.config, tokens, segments, positions=positions)
            logits = cache["logits"][:, 0, :]
            next_ids = np.argmax(logits, axis=-1)
            for row_index in range(len(rows)):
                if finished[row_index]:
                    continue
                token = int(next_ids[row_index])
                rows[row_index].append(token)
                completion[row_index].append(token)
                if token == self.tokenizer.eos_id:
                    finished[row_index] = True
            if all(finished):
                break
        outputs = []
        with self.stats.lock:
            self.stats.requests += len(prompts)
            self.stats.prompt_tokens += sum(prompt_lengths)
        for ids, prompt_len in zip(completion, prompt_lengths):
            text, blocked = self._shield(self.tokenizer.decode(ids))
            finish = "content_filter" if blocked else ("stop" if ids and ids[-1] == self.tokenizer.eos_id else "length")
            with self.stats.lock:
                self.stats.completion_tokens += len(ids)
            outputs.append(
                EngineOutput(
                    text=text,
                    tokens=ids,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(ids),
                    finish_reason=finish,
                )
            )
        return outputs

    def acceptance_rate(self) -> float:
        with self.stats.lock:
            return self.stats.accepted / self.stats.proposed if self.stats.proposed else 0.0
