"""The Cup agent loop.

A deliberately small, debuggable ReAct loop:

    build prompt -> generate (stop at "Observation:") -> parse
      -> if action: run tool, append observation, repeat
      -> if final:  return answer
      -> if neither: nudge once, then bail

Everything that makes this work on a *tiny* model lives in the tolerant parser
(tools.py) and the few-shot prompt (prompts.py), not here. This file stays
boring on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from cup.config import Config
from cup.engine import Engine
from cup.prompts import build_prompt, build_system_prompt
from cup.tools import ParseResult, ToolRegistry, parse


@dataclass
class Step:
    """One think/act/observe cycle, for inspection and debugging."""

    thought_action: str
    tool_name: Optional[str] = None
    tool_args: object = None
    observation: Optional[str] = None


@dataclass
class Result:
    answer: str
    steps: List[Step] = field(default_factory=list)
    stopped_reason: str = "final"  # "final" | "max_steps" | "stuck"


class Agent:
    def __init__(self, engine: Engine, tools: ToolRegistry, config: Optional[Config] = None):
        self.engine = engine
        self.tools = tools
        self.config = config or Config()
        self.system = build_system_prompt(tools, no_think=self.config.no_think)

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def run(self, question: str) -> Result:
        scratchpad = ""
        steps: List[Step] = []
        nudged = False

        for _ in range(self.config.max_steps):
            prompt = build_prompt(self.system, question, scratchpad)
            # Stop the moment a step is complete. A *second* section (a new
            # Thought/Observation/Note/Question on its own line) means the model
            # is rambling past what we asked for — halt server-side to save time.
            completion = self.engine.generate(
                prompt,
                stop=["Observation:", "\nQuestion:", "\nThought:", "\nNote:", "\nObservation"],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ).strip()

            self._log(completion)
            result: ParseResult = parse(completion)

            if result.kind == "final":
                steps.append(Step(thought_action=completion))
                return Result(answer=result.final or "", steps=steps, stopped_reason="final")

            if result.kind == "action":
                tool = self.tools.get(result.tool_name or "")
                if tool is None:
                    available = ", ".join(t.name for t in self.tools) or "(none)"
                    observation = (
                        f"ERROR: unknown tool '{result.tool_name}'. "
                        f"Available tools: {available}."
                    )
                else:
                    observation = tool.run(result.tool_args)

                self._log(f"Observation: {observation}")
                steps.append(
                    Step(
                        thought_action=completion,
                        tool_name=result.tool_name,
                        tool_args=result.tool_args,
                        observation=observation,
                    )
                )
                scratchpad += f"{completion}\nObservation: {observation}\n"
                continue

            # kind == "none": the model didn't follow the format.
            steps.append(Step(thought_action=completion))
            if not nudged:
                nudged = True
                scratchpad += (
                    f"{completion}\n"
                    "Observation: Please reply using 'Action:' + 'Action Input:' "
                    "to use a tool, or 'Final Answer:' to finish.\n"
                )
                continue
            # Already nudged once and still stuck — treat output as the answer.
            return Result(answer=completion, steps=steps, stopped_reason="stuck")

        return Result(
            answer="(stopped: reached max steps without a final answer)",
            steps=steps,
            stopped_reason="max_steps",
        )
