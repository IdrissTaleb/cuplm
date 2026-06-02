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
    ) -> str:
        """Return the model's continuation of `prompt`, stopping at any `stop`
        string (the stop string itself is not included)."""
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
    ) -> str:
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": list(stop),
                "cache_prompt": True,  # reuse KV cache across steps -> much faster
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise ConnectionError(
                f"Could not reach llama.cpp server at {self.base_url}. "
                "Is it running? Start it with `llama-server -m model.gguf --port 8080`."
            ) from exc
        return data.get("content", "")


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
    ) -> str:
        if self._i >= len(self._responses):
            # Fail safe: end the loop rather than hang.
            return "Final Answer: (mock engine ran out of scripted responses)"
        resp = self._responses[self._i]
        self._i += 1
        return resp
