## F_t (failed attempts)
- Must state the action AND the cause based on DOM evidence, not agent belief
- Wrong: "Clicked submit but it didn't work"
- Right: "Clicked Submit at step 8 — form field 'email' still empty per DOM"

## e_t (environment state)
- Pull from ground-truth DOM, NOT from agent's observation text
- If DOM shows out-of-stock, say so even if agent didn't notice
