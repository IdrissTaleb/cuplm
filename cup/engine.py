"""Inference engines for Cup.

The agent talks to an `Engine` through one tiny method: `generate`. This keeps
the agent loop independent of the backend, so we can:

  * run real GGUF models on CPU via llama.cpp (LlamaCppEngine), and
  * test/demo the entire agent loop with zero downloads (MockEngine).

All the heavy compute lives in llama.cpp (C++). Python is just orchestration.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """Anything that can continue a text prompt."""

    def generate(
        self,
        prompt: str,
        stop: Sequence[str],
        max_tokens: int,
        temperature: float,
        grammar: "str | None" = None,
    ) -> str:
        """Return the model's continuation of `prompt`, stopping at any `stop`
        string (the stop string itself is not included). `grammar` is an optional
        GBNF string; engines that can't constrain decoding ignore it."""
        ...


class LlamaCppEngine:
    """Real CPU inference over a GGUF model via llama-cpp-python.

    Install the backend with:  pip install "cup-agent[llama]"
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: "int | None" = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                '  pip install "cup-agent[llama]"\n'
                "Or run with --mock to exercise the agent loop without a model."
            ) from exc

        if not model_path:
            raise ValueError("LlamaCppEngine requires a model_path (a .gguf file).")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    def generate(
        self,
        prompt: str,
        stop: Sequence[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        out = self.llm(
            prompt,
            stop=list(stop),
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        return out["choices"][0]["text"]


class LlamaServerEngine:
    """Talks to a running llama.cpp server over HTTP (stdlib only, no deps).

    This is the recommended backend for weak/legacy CPUs: the official
    llama.cpp server binaries do *runtime* CPU feature detection, so they run
    correctly on machines without AVX-512 (where naive prebuilt Python wheels
    crash with an illegal instruction). The model stays loaded across the whole
    agent loop, so multi-step tasks are fast.

    Start the server yourself, e.g.:
        llama-server -m model.gguf -c 4096 -t 4 --port 8080
    then point Cup at it:
        cup run "..." --server http://localhost:8080
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        stop: Sequence[str],
        max_tokens: int,
        temperature: float,
        grammar: "str | None" = None,
    ) -> str:
        body = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": list(stop),
            "cache_prompt": True,   # reuse KV cache across steps -> much faster
            "repeat_penalty": 1.1,  # tiny models loop; mild penalty curbs it
            "repeat_last_n": 96,
        }
        if grammar:
            body["grammar"] = grammar  # llama.cpp constrains decoding to this GBNF
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # server reachable but errored
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"llama.cpp server returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:  # genuine connection failure
            raise ConnectionError(
                f"Could not reach llama.cpp server at {self.base_url}. "
                "Is it running? Start it with `llama-server -m model.gguf --port 8080`."
            ) from exc
        return data.get("content", "")


def _truncate_at_stops(text: str, stops: Sequence[str]) -> str:
    """Cut `text` at the earliest stop string (exclusive). Mirrors how a server
    applies stop sequences, so the agent loop behaves identically across engines."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


class TransformersEngine:
    """In-process inference of a Hugging Face model (+ optional LoRA adapter).

    This is the bridge for fine-tuning: run a freshly trained `cuplm` adapter and
    benchmark it WITHOUT converting to GGUF first. It also sidesteps the AVX-512
    wheel crash since it uses PyTorch's own kernels.

    Needs torch + transformers (+ peft for adapters):  pip install -r training/requirements.txt
        engine = TransformersEngine("Qwen/Qwen3-0.6B", adapter="training/checkpoints/cuplm-lora")
    """

    def __init__(self, model_id: str, adapter: "str | None" = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        stop: Sequence[str],
        max_tokens: int,
        temperature: float,
        grammar: "str | None" = None,  # ignored: HF path doesn't constrain decoding here
    ) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _truncate_at_stops(text, stop)


class MockEngine:
    """A scripted engine that replays canned completions.

    Lets us prove the agent loop end-to-end (parsing, tool dispatch,
    observation injection, termination) without any model weights. Used by the
    test suite and by `cup --mock`.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    def generate(
        self,
        prompt: str,
        stop: Sequence[str],
        max_tokens: int,
        temperature: float,
        grammar: "str | None" = None,
    ) -> str:
        if self._i >= len(self._responses):
            # Fail safe: end the loop rather than hang.
            return "Final Answer: (mock engine ran out of scripted responses)"
        resp = self._responses[self._i]
        self._i += 1
        return resp
