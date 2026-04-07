# Trajectory JSON Schema

**Status: PLACEHOLDER - update once Rocky shares real trajectory files**

## Expected top-level fields (dummy schema)
- task_id: string
- task_goal: string  
- steps: list of step objects

## Expected step object fields (dummy schema)
- t: int (step number)
- thought: string (agent reasoning)
- action: string (what agent did)
- observation: string (what agent saw - may be stale)
- dom_snapshot: string (ground truth DOM from WebArena - only agent doesn't have this)

## Notes
- dom_snapshot is what the oracle has access to; the task agent only sees observation
- observation and dom_snapshot may diverge, especially at later steps
- Update field names in generate_oracle.py build_prompt() once confirmed
