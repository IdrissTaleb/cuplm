"""GBNF grammar to constrain a model's output to valid Cup ReAct format.

llama.cpp can *guarantee* the model only emits tokens allowed by a grammar.
For a tiny model this is a big reliability win: every response is forced to be
EITHER a well-formed tool call (with a real tool name and a JSON argument
object) OR a final answer. No rambling, no hallucinated tool names, no broken
JSON. Passed to the server as the `grammar` field of /completion.
"""

from __future__ import annotations

from cup.tools import ToolRegistry

# Single line of free text (for Thought / Final Answer) — anything but newline.
# JSON object grammar for Action Input. Kept minimal but correct.
_BASE = r'''
line    ::= [^\n]{1,300}
ws      ::= [ \t]*
object  ::= "{" ws ( pair ( ws "," ws pair )* )? ws "}"
pair    ::= string ws ":" ws value
value   ::= string | number | "true" | "false" | "null"
string  ::= "\"" ( [^"\\] | "\\" . )* "\""
number  ::= "-"? [0-9]+ ( "." [0-9]+ )?
'''


def build_react_grammar(tools: ToolRegistry) -> str:
    """Build a GBNF grammar allowing exactly one tool-call block or one final
    answer, where the tool name must be one of the registered tools."""
    names = [t.name for t in tools]
    if names:
        toolname = " | ".join(f'"{n}"' for n in names)
    else:
        toolname = '""'
    root = (
        'root     ::= toolcall | final\n'
        'toolcall ::= "Thought: " line "\\n" "Action: " toolname "\\n" "Action Input: " object\n'
        'final    ::= ( "Thought: " line "\\n" )? "Final Answer: " line\n'
        f'toolname ::= {toolname}\n'
    )
    return root + _BASE
