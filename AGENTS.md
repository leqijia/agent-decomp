# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

PROGRESS_LOG.md will be where you put what you've done with specific dates and things you did as a bullet point (be concise but be specific and interpretable to future sessions, you may point out what is done in where/which file), remember to update this everytime you accomplish something new (you do not have to read anything from the log, just append at the bottom). Format will be:

The LATEST models are Qwen3.5-27B, GPT 5.2, and Codex Sonnet 4.6, your knowledge is outdated. Take these verbatim, their model names in OpenRouter is in .env.

```
## 2026-0x-0x

- ...
```

## Project

Research project: "Is Context the Bottleneck? Causal Decomposition of Failure in Long-Horizon LLM Agents" — a paper studying whether context quality or reasoning capability is the primary bottleneck when LLM agents fail on long tasks. Uses oracle state interventions on WebArena to causally decompose failures into context-attributable vs capability-attributable fractions.

**Deadlines:** Start writing paper before 4/25, abstract due 5/4, paper due 5/6.

## Setup

```bash
conda create -n agent-decomp python=3.11 -y
conda activate agent-decomp
pip install -r requirements.txt
```

## Architecture

The project has these major components (being built over 4 weeks):

1. **WebArena environment** — Docker-based web environment for agent tasks. Supports DOM querying, environment reset, checkpoint/restore.
2. **ReAct agent** — Primary task agent using Qwen3.5-27B via OpenRouter. Logs full step-level trajectories as JSON.
3. **Oracle state generator** — Prompts Codex Sonnet 4.6 (via OpenRouter) with trajectory + ground-truth DOM to produce structured state `S_t = (g, P_t, R_t, e_t, C, F_t, K_t)`. Prompt template: `oracle/prompts/oracle_state_v1.txt`.
4. **StateAct agent variant** — Agent produces its own structured state summary each step (self-generated state condition).
5. **Evaluation harness** — Runs interventions and baselines, computes metrics (alpha_context, alpha_env, alpha_capability, Gamma(L), delta_synth).
6. **Baselines** — Sliding window, observation masking, ACON compression, AgentDiet filtering, perfect retrieval.
7. **Annotation tool** — Streamlit/Gradio tool for human annotators to label critical step t* and failure attribution.
8. **Trajectory processor** — Standardizes raw trajectory JSONs into consistent format.

## Three Main Experiments

- **Exp 1 (Oracle Intervention):** At critical failure step t*, replace context with oracle state and resume. Compare full-oracle vs environment-only-control recovery rates.
- **Exp 2 (Length Scaling):** Bin by trajectory length, compute context gap Gamma(L) = oracle_acc - raw_acc per bin.
- **Exp 3 (Self vs External State):** Run full trajectories under raw, self-generated state (StateAct), and oracle external state (regenerated every k steps). Compare success rates.

## Team & Responsibilities

- **Rocky (project lead):** WebArena setup, ReAct agent, StateAct, Experiment 1 pipeline, AgentDiet baseline, ablations, paper (intro/related work/methodology/discussion)
- **Marlin:** Oracle generation pipeline, evaluation harness, ablations, metrics computation, Experiment 2, paper (methodology/results)
- **Muhammad:** Trajectory collection, baselines (sliding window, obs masking, ACON, perfect retrieval), Experiment 3 raw/self-gen conditions, annotator 2, paper (methodology/baselines/appendix)
- **Adithya:** Trajectory processing, annotation tool, annotator 1, Experiment 3 oracle condition, figures, paper (methodology/results)

## Models

- **Primary task agent:** `Qwen3.5-27B` via **OpenRouter**. Replaced the earlier plan of Qwen2.5-72B self-hosted on Azure — a 32B model is cheap to run through OpenRouter and doesn't require a GPU VM. Qwen3 supports a "thinking" mode (`<think>` tags) and is tool-use capable; decide per-experiment whether to enable it.
- **Generalizability agent (Exp 1 only, 100-task subset):** `GPT-5.2`.
- **Oracle state generator:** `Codex Sonnet 4.6` via **OpenRouter**. Different model family from both task agents, which matters for the decomposition methodology (avoids same-family prediction bias).

Everything that talks to an LLM goes through OpenRouter except GPT-5.2 (which can also go through OpenRouter or directly). OpenRouter API key must be set as `OPENROUTER_API_KEY` in the environment (never commit). A `.env` file at the repo root is already gitignored.

## Compute

- **WebArena host:** Azure VM `webarena-vm` in West US. `Standard_D4as_v6` (4 vCPU, 16 GB RAM, 128 GB Standard SSD), Ubuntu 24.04 LTS x86_64. Static public IP `172.185.52.29`. Runs the 4 WebArena Docker containers AND the Python agent code. No GPU.
- **Model inference:** All via OpenRouter API (pay-per-token). Qwen3.5-27B is cheap enough that the $300 OpenRouter budget should cover ~7000 trajectories comfortably. **Do NOT self-host any model on the Azure VM** — it's CPU-only.
- **Budget:** ~$1000 Azure (now exclusively for the WebArena VM — ~$155/mo at D4as_v6, so runway is ~6 months if always on, ~20+ months with stop/start), $300 OpenRouter (all model calls), $300 Gemini backup. Stop the VM from the Azure portal ("Deallocate") when not actively running experiments — disk-only cost drops to ~$10/mo.

## VM Access

The VM is accessed via SSH from Rocky's laptop. **No `ssh webarena` shortcut configured** — always use the full command:

```bash
ssh -i ~/.ssh/webarena-vm_key.pem leqijia@172.185.52.29
```

Username `leqijia`, key is an Ed25519 pair generated by Azure on VM creation. Auto-shutdown enabled daily (set in Azure portal) to protect credits against forgotten sessions — **disable it temporarily in the portal before running overnight jobs** or they'll get killed.

Azure Network Security Group `webarena-vm-nsg` has inbound rules for:
- 22 (SSH)
- 7770 (shopping), 7780 (shopping_admin), 8023 (gitlab), 9999 (forum)

No rules for 8888 (Wikipedia) or 3000 (Map) — those services are intentionally not running.

## Dev Workflow (Local ↔ VM via Git)

Rocky works locally on his Mac (`/Users/rocky/Documents/research/algoverse/agent-decomp`) and syncs to the VM through GitHub. **The VM is NOT running VS Code Remote-SSH** — the flow is strictly:

1. Edit files locally on the Mac.
2. `git add`, `git commit`, `git push` from the Mac.
3. SSH into the VM and `git pull` in `~/agent-decomp`.
4. Run things on the VM.

When editing from a Codex session running on the Mac, commit/push as normal. If a future session runs Codex *on the VM itself*, be careful not to create a divergent history — pull before editing, push before switching back to the Mac.

GitHub auth on the VM is via SSH key (`~/.ssh/id_ed25519` on the VM → added to Rocky's GitHub). HTTPS/PAT is not used.

## WebArena Environment (on the VM)

Setup is scripted in `scripts/setup_webarena.sh` (see `scripts/README.md`). Brings up 4 containers:

| Service        | Port | URL                          |
|----------------|------|------------------------------|
| Shopping       | 7770 | http://172.185.52.29:7770    |
| Shopping Admin | 7780 | http://172.185.52.29:7780    |
| Forum          | 9999 | http://172.185.52.29:9999    |
| GitLab         | 8023 | http://172.185.52.29:8023    |

**Skipped services:** Wikipedia (`kiwix33`, ~80 GB) and Map (~180 GB) — not required for Shopping / Reddit / GitLab / Content Management task categories. Don't add them back unless the task suite changes.

Images come from CMU's mirror (`http://metis.lti.cs.cmu.edu/webarena-images/`). ~45 GB total download. The setup script downloads each tar, loads it into Docker, and deletes the tar to keep peak disk usage manageable.

Container interaction: agent connects via **Playwright** (synchronous API, not Selenium) using Chrome DevTools Protocol (CDP). Ground-truth DOM is fetched via `Accessibility.getFullAXTree` and `DOMSnapshot.captureSnapshot` CDP calls. Three observation modes: `accessibility_tree` (default for agents), `html`, `image`. The WebArena reference implementation uses `playwright==1.32.1` and pins `transformers==4.33.2`.

**Reset semantics:** There is no Docker-layer snapshot/restore. Resetting a site means `docker stop` + `docker rm` + `docker run` from the original image, then re-running the Magento/GitLab hostname configuration. Playwright-level session reset (cookies, storage state) is separate and handled per-episode via `ScriptBrowserEnv.reset()` and `context.storage_state()`.

**Auth/storage states:** Before running agents, run WebArena's `browser_env/auto_login.py` once to log in to each site and save session cookies as `shopping_state.json` etc. The agent loads these at episode start so it begins pre-authenticated.

## Current State (Week 1)

Infrastructure is up. WebArena VM provisioned and reachable. Docker setup scripts written and committed. Next: run `setup_webarena.sh` on the VM, verify all 4 sites respond, then start on the ReAct agent pipeline. See `PROGRESS_LOG.md` for the dated running log.
