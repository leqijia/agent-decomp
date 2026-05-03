import os
import sys
import json
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm.client import chat_completion
from llm.config import ORACLE_MODEL

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

WEBARENA_URLS = {
    "shopping":       "http://172.185.52.29:7770",
    "shopping_admin": "http://172.185.52.29:7780",
    "reddit":         "http://172.185.52.29:9999",
    "gitlab":         "http://172.185.52.29:8023",
}
if not API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

# dummy trajectory schema - update field names once Rocky shares real trajectories
DUMMY_TRAJECTORY = [
    {
        "t": 1,
        "thought": "I need to navigate to the shopping page",
        "action": "click('shop_link')",
        "observation": "<div class='page'>Shopping Home</div>"
    },
    {
        "t": 2,
        "thought": "I will search for laptops",
        "action": "type('search_bar', 'laptop')",
        "observation": "<div class='results'>12 laptops found</div>"
    },
    {
        "t": 3,
        "thought": "I need to filter by price",
        "action": "click('sort_by_price')",
        "observation": "<div class='results'>12 laptops sorted by price</div>"
    },
    {
        "t": 4,
        "thought": "I will click the cheapest laptop",
        "action": "click('item_1')",
        "observation": "<div class='product'>Laptop A, $299</div>"
    },
    {
        "t": 5,
        "thought": "I will add it to cart",
        "action": "click('add_to_cart')",
        "observation": "<div class='cart'>1 item added</div>"
    }
]

DUMMY_DOM = "<div class='cart-page'><div class='item'>Laptop A</div><div class='price'>$299</div><div class='stock-status'>In Stock</div><button class='checkout'>Checkout</button></div>"

DUMMY_GOAL = "Find the cheapest laptop on the shopping site and add it to cart"


def get_live_dom(page_url, use_stub=False, use_observation=False, observation=None):
    """
    Fetch the ground-truth accessibility tree for a live WebArena page.

    Args:
        page_url:        the `url` field from a trajectory step
        use_stub:        if True, return a fake accessibility tree (for testing)
        use_observation: if True, return the trajectory step's stored observation
                         as the DOM snapshot (interim fallback until CDP is ready)
        observation:     the `observation` field from the trajectory step;
                         required when use_observation=True

    Returns:
        string — accessibility tree / DOM snapshot

    Full accessibility tree via CDP (Accessibility.getFullAXTree) will replace
    the HTTP GET once Rocky confirms the Playwright method. Until then, use
    use_observation=True to pass the trajectory's stored observation as the
    DOM snapshot.
    """
    if use_stub:
        return (
            "[STUB accessibility tree]\n"
            "RootWebArea 'Page'\n"
            "  heading 'Welcome'\n"
            "  link 'Shop'\n"
            "  button 'Add to cart'\n"
            "  StaticText 'Price: $299'"
        )

    if use_observation:
        if observation is None:
            raise ValueError("observation must be provided when use_observation=True")
        return observation

    # Live HTTP GET — returns raw HTML until CDP method is confirmed with Rocky
    try:
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to WebArena at {page_url}. Is the Docker environment running?")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"Error: Request to {page_url} timed out.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching DOM from {page_url}: {e}")
        sys.exit(1)


def load_prompt(version="v2"):
    path = os.path.join(os.path.dirname(__file__), f"prompts/oracle_state_{version}.txt")
    with open(path, "r") as f:
        return f.read()


def _format_step(s, include_obs):
    if include_obs:
        return (f"Step {s['t']}: thought={s['thought']} | "
                f"action={s['action']} | observation={s['observation']}")
    return f"Step {s['t']}: thought={s['thought']} | action={s['action']}"


def build_prompt(task_goal, trajectory, dom, t, version="v2",
                 observation=None, last_n_obs=3):
    """Render the oracle prompt.

    Prior-step observations are huge accessibility trees and dominate the
    token bill (~96% of trajectory_text at deep t). The oracle's structured
    fields (P_t, R_t, F_t, K_t) only need thought+action history; e_t comes
    from the ground-truth DOM at step t, which is passed separately as
    `dom_snapshot`. We keep full observations on the last `last_n_obs`
    steps before t so the oracle can ground recent agent beliefs against
    what was actually visible.
    """
    template = load_prompt(version)
    obs_floor = t - last_n_obs
    parts = []
    for s in trajectory:
        st = s['t']
        if st > t:
            continue
        if st == t:
            include_obs = False  # avoid duplicating dom_snapshot
        else:
            include_obs = st > obs_floor
        parts.append(_format_step(s, include_obs))
    traj_text = "\n".join(parts)
    dom_snapshot = dom if dom is not None else observation
    return template.format(
        task_goal=task_goal,
        t=t,
        trajectory_text=traj_text,
        dom_snapshot=dom_snapshot,
    )


def build_prompt_split(task_goal, trajectory, dom, t, version="v2",
                       prefix_until_step=None, observation=None,
                       last_n_obs=3):
    """Same as build_prompt but split into (cacheable_prefix, suffix).

    Use for Exp 3 oracle external regeneration where successive calls
    on the same trajectory share the early prefix. Pass
    prefix_until_step=N to put steps 1..N (plus the template instructions
    and goal) into the cached prefix; the per-call suffix carries the
    remaining steps + DOM at step t.

    Same observation-trimming as build_prompt: thought+action only for
    steps further than `last_n_obs` before t; full observation for the
    last `last_n_obs` pre-t steps; step t's DOM is passed separately.
    """
    template = load_prompt(version)
    if prefix_until_step is None:
        # No split — the caller is just asking for the regular prompt.
        return build_prompt(task_goal, trajectory, dom, t, version,
                            observation, last_n_obs), ""

    obs_floor = t - last_n_obs
    cached_steps = [s for s in trajectory
                    if 't' in s and s['t'] <= prefix_until_step]
    later_steps  = [s for s in trajectory
                    if 't' in s and prefix_until_step < s['t'] <= t]

    def _render(steps):
        out = []
        for s in steps:
            st = s['t']
            if st == t:
                include_obs = False  # dom_snapshot carries it
            else:
                include_obs = st > obs_floor
            out.append(_format_step(s, include_obs))
        return "\n".join(out)

    cached_text = _render(cached_steps)
    later_text  = _render(later_steps)
    dom_snapshot = dom if dom is not None else observation

    # The cacheable prefix = template head + goal + early-trajectory body.
    # The suffix = later-trajectory body + DOM at step t + final question.
    full = template.format(
        task_goal=task_goal,
        t=t,
        trajectory_text=cached_text + ("\n" if later_text else ""),
        dom_snapshot=dom_snapshot,
    )
    # Heuristic split: cache everything up to (and including) the cached_text;
    # the suffix is the later trajectory and DOM. We approximate by splitting
    # the rendered template at the dom_snapshot marker.
    marker = dom_snapshot if dom_snapshot else None
    if marker and marker in full and later_text:
        idx = full.index(marker)
        prefix = full[:idx]
        suffix = later_text + "\n" + full[idx:]
        return prefix, suffix
    return full, ""


COST_LOG_PATH = os.path.join(os.path.dirname(__file__), "cost_log.json")

INPUT_TOKEN_RATE = 0.000003   # $3 per 1M input tokens
OUTPUT_TOKEN_RATE = 0.000015  # $15 per 1M output tokens


def _load_cost_log():
    if os.path.exists(COST_LOG_PATH):
        with open(COST_LOG_PATH) as f:
            return json.load(f)
    return {
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "calls": []
    }


def _append_cost_log(trajectory_id, step, prompt_version, input_tokens, output_tokens, cost_usd):
    log = _load_cost_log()
    log["total_calls"] += 1
    log["total_input_tokens"] += input_tokens
    log["total_output_tokens"] += output_tokens
    log["total_cost_usd"] = round(log["total_cost_usd"] + cost_usd, 6)
    log["calls"].append({
        "trajectory_id": trajectory_id,
        "step": step,
        "prompt_version": prompt_version,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    })
    with open(COST_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def call_oracle(prompt, use_stub=True, model=None, cached_prefix=None):
    """
    Call the oracle LLM with the filled prompt.

    Args:
        prompt:        filled prompt string from build_prompt(), OR (when
                       cached_prefix is given) the *suffix* portion of the
                       prompt that should NOT be cached (e.g. the trailing
                       step-t question + DOM).
        use_stub:      if True, return a fake response without hitting any API.
        model:         OpenRouter slug or Gemini model name.
        cached_prefix: optional string. When set and using a Claude model via
                       OpenRouter, this is sent as a separate content block
                       with cache_control={"type": "ephemeral"}, so successive
                       calls on the same trajectory share the cached prefix.
                       Cuts repeat input cost by ~90% on Anthropic models.
    """
    if use_stub:
        print("[STUB] call_oracle called - returning fake response")
        content = json.dumps({
            "g": "Find the cheapest laptop on the shopping site and add it to cart",
            "P_t": ["Navigated to shopping page", "Searched for laptops", "Sorted by price", "Selected cheapest laptop"],
            "R_t": ["Confirm cart and proceed to checkout"],
            "e_t": "Cart page showing Laptop A at $299, item in stock, checkout button visible",
            "C": [],
            "F_t": [],
            "K_t": ["Cheapest laptop is Laptop A", "Price is $299", "Item is in stock"]
        }, indent=2)
        return {"content": content, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "cache_creation_tokens": 0, "cache_read_tokens": 0}

    if cached_prefix is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        # Anthropic prompt caching via OpenRouter: send the stable prefix as a
        # cached content block, then the per-call suffix.
        messages = [{"role": "user", "content": [
            {"type": "text", "text": cached_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]}]

    result = chat_completion(
        messages,
        model=model or ORACLE_MODEL,
        temperature=0.0,
        max_tokens=4096,
    )
    input_tokens = result.prompt_tokens or 0
    output_tokens = result.completion_tokens or 0
    # Cache hits/creates from OpenRouter usage block (Anthropic-style)
    usage = (result.raw or {}).get("usage", {}) or {}
    cache_create = usage.get("cache_creation_input_tokens", 0) or 0
    cache_read   = usage.get("cache_read_input_tokens", 0) or 0
    # Use OpenRouter-reported cost when available (already accounts for
    # cache discount); fall back to estimate.
    if result.cost_usd is not None:
        cost_usd = round(result.cost_usd, 6)
    else:
        # cache reads bill at ~10% of input rate on Anthropic
        regular_in = max(0, input_tokens - cache_read)
        cost_usd = round(
            regular_in * INPUT_TOKEN_RATE
            + cache_read * INPUT_TOKEN_RATE * 0.1
            + cache_create * INPUT_TOKEN_RATE * 1.25
            + output_tokens * OUTPUT_TOKEN_RATE,
            6,
        )
    return {
        "content": result.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
    }


def save_output(trajectory_id, step, prompt_version, response_dict, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    raw_response = response_dict["content"]

    # try to parse as JSON, fall back to raw string
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed = None

    output = {
        "trajectory_id": trajectory_id,
        "step": step,
        "prompt_version": prompt_version,
        "input_tokens": response_dict["input_tokens"],
        "output_tokens": response_dict["output_tokens"],
        "cost_usd": response_dict["cost_usd"],
        "raw_response": raw_response,
        "parsed": parsed,
    }
    path = os.path.join(output_dir, f"{trajectory_id}_t{step}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    # update cost log for real calls (stub calls have 0 tokens)
    if response_dict["input_tokens"] > 0 or response_dict["output_tokens"] > 0:
        _append_cost_log(trajectory_id, step, prompt_version,
                         response_dict["input_tokens"],
                         response_dict["output_tokens"],
                         response_dict["cost_usd"])

    print(f"Saved to {path}")
    return path


if __name__ == "__main__":
    import json as _json

    _TRAJ_PATH = os.path.join(os.path.dirname(__file__), "..", "trajectories", "data", "27.json")
    _T = 16
    _VERSION = "v3"

    print(f"=== Running generate_oracle.py on task 27, step {_T}, prompt {_VERSION} ===\n")

    with open(_TRAJ_PATH) as _f:
        _traj = _json.load(_f)

    _steps = [s for s in _traj["steps"] if "t" in s]
    _step_obj = next(s for s in _steps if s["t"] == _T)
    _dom = _step_obj["observation"]
    _goal = _traj["intent"]
    _task_id = str(_traj["task_id"])

    prompt = build_prompt(_goal, _steps, _dom, t=_T, version=_VERSION)
    print("=== PROMPT PREVIEW (first 1200 chars) ===")
    print(prompt[:1200])
    print("==========================================\n")

    response = call_oracle(prompt, use_stub=False)
    print("=== ORACLE RESPONSE ===")
    print(response["content"])
    print(f"[tokens: {response['input_tokens']} in / {response['output_tokens']} out | cost: ${response['cost_usd']}]")
    print("=======================\n")

    try:
        _parsed = _json.loads(response["content"])
        print("=== P_t ===")
        for item in _parsed.get("P_t", []):
            print(f"  - {item}")
        print("===========")
    except _json.JSONDecodeError:
        print("WARNING: response was not valid JSON")

    save_output(
        trajectory_id=_task_id,
        step=_T,
        prompt_version=_VERSION,
        response_dict=response
    )
