from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

if os.environ.get("MKL_THREADING_LAYER", "").upper() in ("", "INTEL"):
    os.environ["MKL_THREADING_LAYER"] = "GNU"

DEV_WARNING = "DEV RUN — small proxy model, not for research results."


def ensure_utf8(module: str) -> None:
    import os
    import sys
    if os.name != "nt" or os.environ.get("PYTHONUTF8") == "1":
        return
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", module] + sys.argv[1:])

PPL_DEV_MODEL = "EleutherAI/pythia-160m"
PPL_REAL_MODEL = "Qwen/Qwen2.5-1.5B"
RDS_DEV_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RDS_REAL_MODEL = "meta-llama/Llama-2-7b-hf"
QUALITY_DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUALITY_REAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

RESULTS_DIR = "audit/results"
DEV_RESULTS_DIR = "audit/results/dev"


def results_base(dev: bool) -> str:
    return DEV_RESULTS_DIR if dev else RESULTS_DIR


def model_slug(model: str) -> str:
    return model.split("/")[-1].lower().replace("_", "-")


GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<start_of_turn>' + (message['role'] if message['role'] != 'assistant' else 'model') "
    "+ '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)


LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def ensure_chat_template(tok, model_id: str):
    if getattr(tok, "chat_template", None):
        return tok
    mid = model_id.lower()
    if "gemma" in mid:
        tok.chat_template = GEMMA_CHAT_TEMPLATE
    elif "llama" in mid:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def announce_dev(dev: bool, logger: logging.Logger) -> None:
    if dev:
        logger.warning(DEV_WARNING)


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def device_str() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VramSampler:

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak_mib = None
        self.runtime_s = None
        self._stop = threading.Event()
        self._t0 = None
        self._thread = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=5,
                ).decode().strip().splitlines()
                used = max(int(x) for x in out if x.strip())
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramSampler":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.runtime_s = round(time.time() - self._t0, 1)


def run_meta(model: str, dev: bool, sampler: "VramSampler | None" = None,
             extra: dict | None = None) -> dict:
    meta = {
        "model": model,
        "dev": dev,
        "device": device_str(),
        "gpu_name": gpu_name(),
        "runtime_s": sampler.runtime_s if sampler else None,
        "vram_peak_mib": sampler.peak_mib if sampler else None,
    }
    if extra:
        meta.update(extra)
    return meta
