import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
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


def load_prompt(version="v2"):
    path = os.path.join(os.path.dirname(__file__), f"prompts/oracle_state_{version}.txt")
    with open(path, "r") as f:
        return f.read()


def build_prompt(task_goal, trajectory, dom, t, version="v2"):
    template = load_prompt(version)
    # update these field names once Rocky shares real trajectory schema
    traj_text = "\n".join([
        f"Step {s['t']}: thought={s['thought']} | action={s['action']} | observation={s['observation']}"
        for s in trajectory if s['t'] <= t
    ])
    return template.format(
        task_goal=task_goal,
        t=t,
        trajectory_text=traj_text,
        dom_snapshot=dom
    )


def call_oracle(prompt, use_stub=True):
    if use_stub:
        print("[STUB] call_oracle called - returning fake response")
        return json.dumps({
            "g": "Find the cheapest laptop on the shopping site and add it to cart",
            "P_t": ["Navigated to shopping page", "Searched for laptops", "Sorted by price", "Selected cheapest laptop"],
            "R_t": ["Confirm cart and proceed to checkout"],
            "e_t": "Cart page showing Laptop A at $299, item in stock, checkout button visible",
            "C": [],
            "F_t": [],
            "K_t": ["Cheapest laptop is Laptop A", "Price is $299", "Item is in stock"]
        }, indent=2)

    # real API call - uncomment when OpenRouter credits are available
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "anthropic/claude-sonnet-4-6",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("Error: Invalid API key.")
        elif response.status_code == 402:
            print("Error: Insufficient credits on OpenRouter account.")
        else:
            print(f"HTTP error {response.status_code}: {e}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
        sys.exit(1)


def save_output(trajectory_id, step, prompt_version, raw_response, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # try to parse as JSON, fall back to raw string
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed = None

    output = {
        "trajectory_id": trajectory_id,
        "step": step,
        "prompt_version": prompt_version,
        "raw_response": raw_response,
        "parsed": parsed
    }
    path = os.path.join(output_dir, f"{trajectory_id}_t{step}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {path}")
    return path


if __name__ == "__main__":
    print("=== Running generate_oracle.py with dummy data ===\n")

    prompt = build_prompt(DUMMY_GOAL, DUMMY_TRAJECTORY, DUMMY_DOM, t=5)
    print("=== PROMPT PREVIEW (first 1200 chars) ===")
    print(prompt[:1200])
    print("==========================================\n")

    response = call_oracle(prompt, use_stub=False)
    print("=== ORACLE RESPONSE ===")
    print(response)
    print("=======================\n")

    save_output(
        trajectory_id="dummy_task_001",
        step=5,
        prompt_version="v2",
        raw_response=response
    )
