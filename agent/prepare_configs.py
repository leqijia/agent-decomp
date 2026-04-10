"""Generate per-task config files from upstream test.raw.json.

Reads references/webarena/config_files/test.raw.json, substitutes the
__SITE__ URL templates with the real URLs from .env, filters to the sites
we actually run (Shopping, Shopping Admin, GitLab, Reddit), and writes
individual JSON files to config_files/<task_id>.json.

Usage:
    python -m agent.prepare_configs [--out-dir config_files]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_RAW_CONFIG = Path(__file__).parent.parent / "references" / "webarena" / "config_files" / "test.raw.json"

_URL_MAP = {
    "__SHOPPING__": os.environ.get("SHOPPING", ""),
    "__SHOPPING_ADMIN__": os.environ.get("SHOPPING_ADMIN", ""),
    "__GITLAB__": os.environ.get("GITLAB", ""),
    "__REDDIT__": os.environ.get("REDDIT", ""),
    "__MAP__": os.environ.get("MAP", ""),
}

_OUR_SITES = {"shopping", "shopping_admin", "reddit", "gitlab"}


def substitute_urls(config: dict) -> dict:
    """Replace __SITE__ placeholders in start_url and anywhere else."""
    raw = json.dumps(config)
    for placeholder, url in _URL_MAP.items():
        raw = raw.replace(placeholder, url)
    return json.loads(raw)


def prepare(out_dir: Path, site_filter: set[str] | None = None) -> list[Path]:
    """Write per-task configs and return list of written paths."""
    with open(_RAW_CONFIG) as f:
        all_configs = json.load(f)

    sites = site_filter or _OUR_SITES
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for cfg in all_configs:
        task_sites = set(cfg.get("sites", []))
        if not task_sites.issubset(sites):
            continue

        cfg = substitute_urls(cfg)
        path = out_dir / f"{cfg['task_id']}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-task WebArena configs.")
    parser.add_argument("--out-dir", default="config_files", help="Output directory")
    args = parser.parse_args()

    written = prepare(Path(args.out_dir))
    print(f"Wrote {len(written)} task configs to {args.out_dir}/")


if __name__ == "__main__":
    main()
