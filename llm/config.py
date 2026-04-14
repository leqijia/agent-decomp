"""Central model configuration.

This is the single source of truth for which model serves each role.
All other modules import defaults from here rather than hardcoding slugs
or duplicating os.environ.get() calls.

To switch a model: edit the relevant env var in .env (or export it).
The fallback values below are the production defaults for the paper.

All models run through OpenRouter (OPENROUTER_API_KEY). The Gemini API
path in llm/client.py is preserved for reference but is not used in
production — OpenRouter serves Gemini models too, under one key.

Roles
-----
AGENT_MODEL            Primary task agent (ReAct loop, trajectory collection).
                       Qwen3.5-27B via OpenRouter — cheap enough for ~7000
                       trajectories within the $300 OpenRouter budget.

ORACLE_MODEL           Oracle state generator (Exp 1/2/3 context injection).
                       Claude Sonnet 4.6 via OpenRouter — different model
                       family from the task agent, which matters for the
                       decomposition methodology.

GENERALIZABILITY_MODEL Exp 1 generalizability condition (100-task subset).
                       GPT-4.5 via OpenRouter.

ACON_MODEL             ACON compression baseline LLM.
                       Defaults to same as ORACLE_MODEL (Claude Sonnet).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

AGENT_MODEL: str = os.environ.get("AGENT_MODEL", "qwen/qwen3.5-27b")
ORACLE_MODEL: str = os.environ.get("ORACLE_MODEL", "anthropic/claude-sonnet-4-6")
GENERALIZABILITY_MODEL: str = os.environ.get("GENERALIZABILITY_MODEL", "openai/gpt-4.5")
ACON_MODEL: str = os.environ.get("ACON_MODEL", "anthropic/claude-sonnet-4-6")
