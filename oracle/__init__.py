"""
Oracle state generation pipeline.

Exposes the three functions needed to generate and save oracle states
during live Experiment 3 agent runs: build_prompt, call_oracle, save_output.
"""

from oracle.generate_oracle import build_prompt, call_oracle, save_output

__all__ = ["build_prompt", "call_oracle", "save_output"]
