"""CoT prompt constructor, simplified from upstream WebArena.

Upstream's `agent/prompts/prompt_constructor.py` supports multiple providers
(OpenAI chat, OpenAI completion, HuggingFace Llama-2) and pulls in an
`llms/` package we do not vendor. We only ever talk to OpenRouter in the
chat-completions format, so this is a thin rewrite that:

  1. Loads the same JSON prompt template format upstream uses.
  2. Builds an OpenRouter chat-messages list (system intro, few-shot pairs,
     user turn with observation + url + objective + previous action).
  3. Truncates the current observation to max_obs_length tokens using the
     cl100k_base tokenizer (approximates any chat model's tokenizer well
     enough for budget bookkeeping; trajectories/SPEC.md stores the
     post-truncation text so it is reproducible).
  4. Extracts the action from the model response using the
     `action_splitter` from the template's meta_data.

Keeping the JSON template format compatible with upstream means we can
drop in any of upstream's p_cot_* templates later without touching this
file — only the JSON changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from webarena.browser_env.actions import ActionParsingError

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class PromptTemplate:
    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PromptTemplate":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            intro=raw["intro"],
            examples=[tuple(e) for e in raw["examples"]],
            template=raw["template"],
            meta_data=raw["meta_data"],
        )


def truncate_observation(obs: str, max_obs_length: int) -> str:
    """Truncate an accessibility-tree observation to max_obs_length tokens.

    `max_obs_length` matches upstream's --max_obs_length flag and is also
    what trajectories/SPEC.md records per episode. The truncated string is
    what the agent actually sees and what we persist, so the oracle and Exp
    1 replay reproduce the agent's exact view.
    """
    if max_obs_length <= 0:
        return obs
    token_ids = _TOKENIZER.encode(obs)
    if len(token_ids) <= max_obs_length:
        return obs
    return _TOKENIZER.decode(token_ids[:max_obs_length])


def build_messages(
    template: PromptTemplate,
    *,
    objective: str,
    observation: str,
    url: str,
    previous_action: str,
) -> list[dict[str, str]]:
    """Assemble the OpenRouter chat-messages list from a template.

    The upstream format uses role=system for the intro and the few-shot
    pairs (with the example_user / example_assistant name hack). We
    collapse that to plain system + alternating user/assistant, because
    OpenRouter's providers do not all honor the name field, and the
    alternating form is semantically identical for few-shot prompting.
    """
    user_turn = template.template.format(
        objective=objective,
        url=url,
        observation=observation,
        previous_action=previous_action,
    )
    # Fail loudly if any format key slipped through unreplaced.
    for k in template.meta_data.get("keywords", []):
        assert f"{{{k}}}" not in user_turn, f"unsubstituted keyword: {k}"

    messages: list[dict[str, str]] = [{"role": "system", "content": template.intro}]
    for ex_user, ex_assistant in template.examples:
        messages.append({"role": "user", "content": ex_user})
        messages.append({"role": "assistant", "content": ex_assistant})
    messages.append({"role": "user", "content": user_turn})
    return messages


def extract_action(template: PromptTemplate, response: str) -> str:
    """Pull the action string out of the model's response.

    The CoT template wraps the action in the `action_splitter` (```), so
    this looks for the first match of splitter((.|\\n)*?)splitter. Raises
    ActionParsingError if no match is found; the ReAct loop catches that
    and counts it toward the early-stop parse-failure threshold.
    """
    splitter = template.meta_data["action_splitter"]
    pattern = rf"{re.escape(splitter)}((.|\n)*?){re.escape(splitter)}"
    match = re.search(pattern, response)
    if match is None:
        raise ActionParsingError(
            f'Could not find action between {splitter!r} markers in response'
        )
    return match.group(1).strip()
