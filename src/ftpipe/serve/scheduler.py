"""Batch scheduler around the generation engine."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Any


class BatchScheduler:
    def __init__(self, engine: Any, max_batch: int = 8, window_ms: int = 5):
        self.engine = engine
        self.max_batch = max_batch
        self.window_ms = window_ms
        self.queue: queue.Queue = queue.Queue()
        self.stats = engine.stats
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, prompt: str, max_new_tokens: int) -> Future:
        future: Future = Future()
        with self.stats.lock:
            self.stats.waiting += 1
        self.queue.put((prompt, max_new_tokens, future))
        return future

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                first = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            items = [first]
            deadline = time.monotonic() + self.window_ms / 1000.0
            while len(items) < self.max_batch:
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    items.append(self.queue.get(timeout=timeout))
                except queue.Empty:
                    break
            with self.stats.lock:
                self.stats.waiting = max(0, self.stats.waiting - len(items))
            admitted = []
            for item in items:
                try:
                    self.engine.check_prompt(item[0])
                    admitted.append(item)
                except Exception as exc:
                    item[2].set_exception(exc)
            if not admitted:
                continue
            try:
                outputs = self.engine.generate_many(
                    [item[0] for item in admitted], max_new_tokens=admitted[0][1]
                )
            except Exception as exc:
                for item in admitted:
                    item[2].set_exception(exc)
                continue
            for output, item in zip(outputs, admitted):
                item[2].set_result(output)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
