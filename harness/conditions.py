from typing import Protocol, runtime_checkable

VALID_CONDITIONS = {
    "raw",
    "sliding_window",
    "observation_masking",
    "acon",
    "agentdiet",
    "perfect_retrieval",
    "env_only",
    "self_generated",
    "oracle_external",
}


@runtime_checkable
class ContextPolicy(Protocol):
    """
    Interface for context transformation policies applied before each agent step.
    Implementations receive the raw trajectory and return a (possibly modified)
    context string to inject into the agent prompt.
    """

    def apply(self, trajectory: list[dict], step: int, **kwargs) -> str:
        """Transform trajectory context for a given step. Returns context string."""
        ...

    def name(self) -> str:
        """Return a short identifier for this policy (used in logs and filenames)."""
        ...


# Maps condition name -> (agent_variant, policy_name)
# agent_variant: which agent class/config to instantiate
# policy_name:   which ContextPolicy implementation to apply
CONDITION_REGISTRY: dict[str, tuple[str, str]] = {
    "raw":                ("standard",     "identity"),
    "sliding_window":     ("standard",     "sliding_window"),
    "observation_masking":("standard",     "observation_masking"),
    "acon":               ("acon",         "identity"),
    "agentdiet":          ("agentdiet",    "identity"),
    "perfect_retrieval":  ("standard",     "perfect_retrieval"),
    "env_only":           ("standard",     "env_only"),
    "self_generated":     ("standard",     "self_generated"),
    "oracle_external":    ("standard",     "oracle_external"),
}
