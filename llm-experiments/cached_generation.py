from __future__ import annotations
import torch
"""
`cached_generate` – persistent cache ignoring `n` and `max_tokens`
=================================================================
This wrapper stores **every** sequence returned by **vLLM** once and re‑uses it
on future calls – irrespective of how many tokens the model produced.  If you
request *T* tokens and the model stops early, that shorter text still counts
as a valid sample and is cached (exactly what vLLM gave you).  A prompt‑model
pair therefore has *one* cache file,
and subsequent calls never repeat the same work.

**What's cached?**
* `text`, `token_ids`, `logprobs`
* `meta`: model name, dtype, full sampling params (including the *requested*
  `max_tokens`), sequence index, and generation timestamp

**Identity rules (file fingerprint)**
* Ignores `SamplingParams.n` and `SamplingParams.max_tokens` so that different
  values of either share the same cache.

**Call semantics**
1. Load all cached sequences (length doesn't matter).
2. If fewer than the requested `n` samples exist, generate the shortfall with
   the given `sampling_params` **once**, append them to disk, and return the
   combined list.
3. Returned objects emulate vLLM's `RequestOutput` API (`.outputs[i].text`, …).

Because early‑stopped generations are acceptable, we no longer treat token
length specially – simpler, faster, and closer to vLLM's contract.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Sequence, Tuple, Union
import threading
import queue
from collections import defaultdict

# ---------------------------------------------------------------------------
# File locking helper
# ---------------------------------------------------------------------------
try:
    from filelock import FileLock  # type: ignore

    def _file_lock(path: str):
        return FileLock(path + ".lock")
except ImportError:  # fallback spin‑lock
    @contextmanager
    def _file_lock(path: str):  # type: ignore
        lock = path + ".lock"
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass

# ---------------------------------------------------------------------------
# Configuration & hash helpers (ignores n AND max_tokens)
# ---------------------------------------------------------------------------
_CACHE_ROOT = None # a folder to cache input/output pairs
if _CACHE_ROOT is None:
    raise ValueError("CACHE_ROOT must be set in cached_generation.py!")

def _sampling_params_fingerprint(sp) -> str:
    payload = {k: v for k, v in sp.__dict__.items() if k not in {"n", "max_tokens"}}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]

def _prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# JSON‑serialisable line format
# ---------------------------------------------------------------------------
@dataclass
class _StoredSequence:
    text: str
    token_ids: List[int] | None = None
    logprobs: Any | None = None
    meta: Dict[str, Any] | None = None

    @classmethod
    def from_vllm(cls, seq_out: Any, meta: Dict[str, Any]):
        return cls(
            text=seq_out.text,
            token_ids=getattr(seq_out, "token_ids", None),
            logprobs=getattr(seq_out, "logprobs", None),
            meta=meta,
        )

    def to_vllm(self, max_tokens: int | None = None):
        return type("CachedSequenceOutput", (), {
            "text": self.text,
            "token_ids": self.token_ids[:max_tokens],
            "logprobs": (self.logprobs[:max_tokens] if self.logprobs is not None else None),
        })()

# ---------------------------------------------------------------------------
# Dedicated writer thread for asynchronous batched writes.
# ---------------------------------------------------------------------------
class CacheWriter:
    def __init__(self, threshold: int = 10000):
        """
        threshold: number of JSON lines to batch into one write.
        """
        self.threshold = threshold
        self.buffers = defaultdict(list)  # Key: cache_file, Value: list of JSON lines
        self.locks = defaultdict(threading.Lock)
        self.write_queue = queue.Queue()
        # Start a single dedicated writer thread.
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def schedule_write(self, stored: _StoredSequence, cache_file: str):
        """
        Buffer a stored sequence for asynchronous writing.
        When the buffer for a cache file reaches the threshold, send the batch over
        to the dedicated writer thread.
        """
        line = json.dumps(asdict(stored))
        with self.locks[cache_file]:
            self.buffers[cache_file].append(line)
            if len(self.buffers[cache_file]) >= self.threshold:
                lines_to_write = self.buffers[cache_file]
                self.buffers[cache_file] = []
                self.write_queue.put((cache_file, lines_to_write))

    def flush_buffer(self, cache_file: str):
        """Flush remaining buffered writes for a given cache file."""
        with self.locks[cache_file]:
            if self.buffers[cache_file]:
                lines_to_write = self.buffers[cache_file]
                self.buffers[cache_file] = []
                self.write_queue.put((cache_file, lines_to_write))

    def _writer_worker(self):
        """The dedicated writer thread that pops batches from the queue and writes them."""
        while True:
            task = self.write_queue.get()
            if task is None:
                break
            cache_file, lines = task
            with _file_lock(cache_file):
                with open(cache_file, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            self.write_queue.task_done()

    def stop(self):
        """Stops the writer thread gracefully."""
        self.write_queue.put(None)
        self.writer_thread.join()


# ---------------------------------------------------------------------------
# Main entry‑point: cached_generate
# ---------------------------------------------------------------------------
def cached_generate(llm, prompts: Union[str, Sequence[str]], sampling_params, model=None):
    # Create a single global cache writer instance.
    cache_writer = CacheWriter()
    """
    Disk‑cached replacement for `llm.generate` using a single file per model.

    Caching is based on a combination of a prompt fingerprint and a sampling-parameters
    fingerprint (ignoring n and max_tokens). Extended sequences are replaced by newer
    versions (using deduplication upon load), and complete sequences (i.e., with enough tokens)
    are prioritized over incomplete ones.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    model = model or getattr(llm, "model_name", "unknown-model")
    dtype = str(getattr(llm, "dtype", torch.bfloat16))
    sp_fp = _sampling_params_fingerprint(sampling_params)
    cache_file = os.path.join(_CACHE_ROOT, f"{model}__{dtype}.jsonl")
    print("loading cache from", cache_file)

    # 1. Load the entire cache file and build a deduplicated dictionary.
    cache_dict: Dict[Tuple[str, str], List[_StoredSequence]] = {}
    if os.path.exists(cache_file):
        records_by_key: Dict[Tuple[str, str, int], _StoredSequence] = {}
        with _file_lock(cache_file):
            with open(cache_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip any corrupted lines.
                    stored = _StoredSequence(**data)
                    prompt_fp = stored.meta.get("prompt_fp")
                    sp_fp_line = stored.meta.get("sp_fp")
                    seq_index = stored.meta.get("sequence_index")
                    if prompt_fp is None or sp_fp_line is None or seq_index is None:
                        continue
                    key_full = (prompt_fp, sp_fp_line, seq_index)

                    # key_full = (prompt_fp, seq_index)
                    # Replace if newer (by timestamp) than any existing one.
                    if key_full in records_by_key:
                        if stored.meta.get("timestamp", 0) > records_by_key[key_full].meta.get("timestamp", 0):
                            records_by_key[key_full] = stored
                    else:
                        records_by_key[key_full] = stored
        for (prompt_fp, sp_fp_line, _), stored in records_by_key.items():
            # cache_dict.setdefault((prompt_fp, sp_fp_line), []).append(stored)

            cache_dict.setdefault((prompt_fp), []).append(stored)
    print("cache loaded")

    target_max = getattr(sampling_params, "max_tokens", None)
    target_n = sampling_params.n or 1
    per_prompt_cache: List[List[_StoredSequence]] = []
    prompt_fps: List[str] = []
    test = []
    for p in prompts:
        p_fp = _prompt_fingerprint(p)
        prompt_fps.append(p_fp)
        # key = (p_fp, sp_fp)
        key = (p_fp)
        seqs = cache_dict.get(key, [])
        complete = []
        for seq in seqs:
            if len(seq.token_ids) >= target_max:
                complete.append(seq)
            else:
                if len(seq.token_ids) not in test:
                    test.append(len(seq.token_ids))

        per_prompt_cache.append(complete)

    # Determine how many additional generations are needed per prompt.
    needs = [max(0, target_n - len(seqs)) for seqs in per_prompt_cache]
    if llm is None and any(needs):
        print("llm is None and needs", sum(needs))
        return False

    # 2. Batch generation for prompts that need additional outputs.
    indices_to_generate = [i for i, need in enumerate(needs) if need > 0]
    if indices_to_generate:
        generation_prompts = [prompts[i] for i in indices_to_generate]
        generation_needs = [needs[i] for i in indices_to_generate]
        max_needed = max(generation_needs)
        sp_clone = sampling_params.clone()
        sp_clone.n = max_needed
        new_outputs = llm.generate(generation_prompts, sp_clone)
        print("new_outputs", len(new_outputs))
        print("indices_to_generate", len(indices_to_generate))
        for j, idx in enumerate(indices_to_generate):
            req_out = new_outputs[j]
            p_fp = prompt_fps[idx]
            key = (p_fp, sp_fp)
            for seq_idx, seq_out in enumerate(req_out.outputs):  # type: ignore[attr-defined]
                meta = {
                    "timestamp": time.time(),
                    "model": model,
                    "dtype": dtype,
                    "sampling_params": {
                        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                        for k, v in sampling_params.__dict__.items()
                    },
                    "sequence_index": seq_idx,
                    "prompt_fp": p_fp,
                    "sp_fp": sp_fp,
                    "finish_reason": getattr(seq_out, "finish_reason", None),
                }
                stored = _StoredSequence.from_vllm(seq_out, meta)
                cache_writer.schedule_write(stored, cache_file)
                cache_dict.setdefault(key, []).append(stored)
                per_prompt_cache[idx].append(stored)

    cache_writer.flush_buffer(cache_file)
    # 3. Assemble outputs while prioritizing complete sequences.
    outputs_by_prompt = []
    extension_pairs: List[_StoredSequence] = []
    for i, seqs in enumerate(per_prompt_cache):
        complete = []
        incomplete = []
        for s in seqs:
            # A sequence is "complete" if max_tokens is undefined
            # or it has reached (or exceeded) the desired token count.
            if target_max is None or len(s.token_ids) >= target_max-1:
                complete.append(s)
            else:
                # print(target_max-len(s.token_ids))
                incomplete.append(s)
        chosen = complete[:target_n]
            
        if len(complete) < target_n:
            print("not enough complete sequences", len(complete), "needed", target_n)
            needed = target_n - len(complete)
            # chosen = complete + incomplete[:needed]
            extension_pairs.extend(incomplete[:needed])
        outputs_by_prompt.append((prompts[i], chosen))


    if extension_pairs and needed:
        assert False
        # if llm is None:
        #     return False
    
    results = []
    for prompt, chosen in outputs_by_prompt:
        extended_chosen = [s.to_vllm(max_tokens=target_max) for s in chosen]
        results.append(
            type("CachedRequestOutput", (), {"prompt": prompt, "outputs": extended_chosen})()
        )
    cache_writer.flush_buffer(cache_file)
    cache_writer.stop()
    return results
