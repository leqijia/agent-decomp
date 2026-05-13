"""Preflight checks for the one-command remaining-experiments runner.

This is intentionally lightweight but live: it catches the common failure
classes before `scripts/run_remaining.sh` spends hours writing crash artifacts:
missing dependencies, missing OpenRouter key, unusable model/key quota, missing
task/config artifacts, and unavailable WebArena sites.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IMPORTS = [
    "dotenv",
    "requests",
    "tiktoken",
    "numpy",
    "playwright",
]
SITE_URLS = {
    "shopping": "http://172.185.52.29:7770",
    "shopping_admin": "http://172.185.52.29:7780",
    "gitlab": "http://172.185.52.29:8023",
    "forum": "http://172.185.52.29:9999",
}


def _fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_imports() -> None:
    missing = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        _fail(
            "missing Python dependencies: "
            + ", ".join(missing)
            + ". Run `pip install -r requirements.txt` in the VM environment."
        )
    print("dependencies: ok")


def check_artifacts() -> None:
    required_paths = [
        ROOT / "config_files",
        ROOT / "trajectories" / "data",
        ROOT / "experiments" / "exp3" / "raw",
        ROOT / "annotations",
        ROOT / "oracle" / "outputs" / "tstar",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required_paths if not p.exists()]
    if missing:
        _fail("missing required project artifacts: " + ", ".join(missing))

    raw_count = len(list((ROOT / "trajectories" / "data").glob("*.json")))
    config_count = len(list((ROOT / "config_files").glob("*.json")))
    tstar_count = len(list((ROOT / "oracle" / "outputs" / "tstar").glob("*/*.json")))
    if raw_count == 0 or config_count == 0:
        _fail("raw trajectories/config files are empty; nothing useful can run.")
    print(f"artifacts: ok ({raw_count} raw trajectories, {config_count} configs, {tstar_count} t* oracles)")


def check_sites(timeout_s: float = 5.0, attempts: int = 3) -> None:
    import requests

    down = []
    for name, url in SITE_URLS.items():
        ok = False
        last_err = ""
        for _ in range(attempts):
            try:
                resp = requests.get(url, timeout=timeout_s)
                if resp.status_code < 500:
                    ok = True
                    break
                last_err = f"HTTP {resp.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(2)
        if not ok:
            down.append(f"{name} ({url}: {last_err})")
    if down:
        _fail(
            "WebArena sites are not reachable: "
            + "; ".join(down)
            + ". Start/fix Docker containers before running experiments."
        )
    print("webarena sites: ok")


def check_openrouter_models(skip_gpt52: bool) -> None:
    from llm.config import AGENT_MODEL, GENERALIZABILITY_MODEL, ORACLE_MODEL
    from llm.openrouter import OpenRouterError, chat_completion

    if not os.environ.get("OPENROUTER_API_KEY"):
        _fail("OPENROUTER_API_KEY is not set in the environment or .env.")

    models = [("agent", AGENT_MODEL), ("oracle", ORACLE_MODEL)]
    if not skip_gpt52:
        models.append(("generalizability", GENERALIZABILITY_MODEL))

    seen: set[str] = set()
    for role, model in models:
        if model in seen:
            continue
        seen.add(model)
        try:
            chat_completion(
                [{"role": "user", "content": "Reply with OK."}],
                model=model,
                temperature=0.0,
                max_tokens=4,
                timeout_s=45,
            )
        except OpenRouterError as e:
            _fail(f"OpenRouter preflight failed for {role} model `{model}`: {e}")
        print(f"openrouter {role} model: ok ({model})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpt52", action="store_true")
    parser.add_argument(
        "--skip-sites",
        action="store_true",
        help="Skip WebArena HTTP checks. Use only when networking is intentionally blocked.",
    )
    parser.add_argument(
        "--skip-openrouter",
        action="store_true",
        help="Skip live OpenRouter model/key checks.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    check_imports()
    from dotenv import load_dotenv
    load_dotenv()

    check_artifacts()
    if not args.skip_sites:
        check_sites()
    if not args.skip_openrouter:
        check_openrouter_models(skip_gpt52=args.skip_gpt52)

    print("preflight: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
