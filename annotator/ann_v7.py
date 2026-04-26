from __future__ import annotations

import html
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


_REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY_DIR = _REPO_ROOT / "trajectories" / "data"
ANNOTATION_DIR = _REPO_ROOT / "annotations"
CONFIG_DIR = _REPO_ROOT / "config_files"


@dataclass
class AnnotationRecord:
    annotator_id: str
    trajectory_id: str
    t_star_step: int
    failure_classification: Optional[str]
    timestamp: str
    notes: str = ""
    # True: t_star_step is trajectory step t (1..N) matching the JSON "t" field. False/legacy: 0-based list index.
    t_star_is_trajectory_t: bool = True


def ensure_dirs() -> None:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)


def list_trajectory_files() -> List[Path]:
    ensure_dirs()
    return sorted(TRAJECTORY_DIR.glob("*.json"))


def get_trajectory_id(data: Dict[str, Any], fallback: str) -> str:
    if data.get("trajectory_id") is not None:
        return str(data["trajectory_id"])
    if data.get("task_id") is not None:
        return f"task_{data['task_id']}"
    return fallback


def list_trajectory_ids() -> List[str]:
    ids: List[str] = []
    for path in list_trajectory_files():
        try:
            data = json.loads(path.read_text())
            ids.append(get_trajectory_id(data, path.stem))
        except Exception:
            ids.append(path.stem)
    return ids


def get_trajectory_path_by_id(trajectory_id: str) -> Optional[Path]:
    for path in list_trajectory_files():
        try:
            data = json.loads(path.read_text())
            if get_trajectory_id(data, path.stem) == trajectory_id:
                return path
        except Exception:
            if path.stem == trajectory_id:
                return path
    return None


def load_trajectory(trajectory_id: str) -> Dict[str, Any]:
    path = get_trajectory_path_by_id(trajectory_id)
    if path is None:
        raise FileNotFoundError(f"Trajectory not found: {trajectory_id}")
    return json.loads(path.read_text())


def load_config_for_trajectory(trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_id = trajectory.get("task_id")
    if task_id is None:
        return None
    path = CONFIG_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def annotation_path(annotator_id: str, trajectory_id: str, create_dir: bool = False) -> Path:
    annotator_dir = ANNOTATION_DIR / annotator_id.strip()
    if create_dir:
        annotator_dir.mkdir(parents=True, exist_ok=True)
    return annotator_dir / f"{trajectory_id}.json"


def load_annotation(annotator_id: str, trajectory_id: str) -> Optional[Dict[str, Any]]:
    annotator_id = annotator_id.strip()
    if not annotator_id or not trajectory_id:
        return None
    path = annotation_path(annotator_id, trajectory_id, create_dir=False)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_annotation(record: AnnotationRecord) -> Path:
    out_path = annotation_path(record.annotator_id, record.trajectory_id, create_dir=True)
    out_path.write_text(json.dumps(asdict(record), indent=2))
    return out_path


def count_completed_for_annotator(annotator_id: str) -> Tuple[int, int]:
    total = len(list_trajectory_files())
    annotator_id = annotator_id.strip()
    if not annotator_id:
        return 0, total
    annotator_dir = ANNOTATION_DIR / annotator_id
    completed = len(list(annotator_dir.glob("*.json"))) if annotator_dir.exists() else 0
    return completed, total


def build_peer_status() -> Dict[str, List[str]]:
    """Return {trajectory_id: sorted list of annotator_ids who have saved an annotation}."""
    ensure_dirs()
    status: Dict[str, List[str]] = {}
    for annot_dir in ANNOTATION_DIR.iterdir():
        if not annot_dir.is_dir():
            continue
        for f in annot_dir.glob("*.json"):
            status.setdefault(f.stem, []).append(annot_dir.name)
    for tid in status:
        status[tid].sort()
    return status


def build_dropdown_choices() -> List[Tuple[str, str]]:
    peer = build_peer_status()
    choices: List[Tuple[str, str]] = []
    for tid in list_trajectory_ids():
        annotators = peer.get(tid, [])
        label = f"{tid}  \u2022  [{', '.join(annotators)}]" if annotators else tid
        choices.append((label, tid))
    return choices


def get_steps(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    return trajectory.get("steps", [])


def get_step_number(step: Dict[str, Any], list_index: int) -> int:
    """Trajectory step id (WebArena 't' / step_id). Falls back to 1-based list position when missing."""
    if step.get("step_id") is not None:
        return int(step["step_id"])
    if step.get("t") is not None:
        return int(step["t"])
    return list_index + 1


def build_expected_answer_md(trajectory: Dict[str, Any]) -> str:
    unavailable = "## Expected Answer\n_Reference answer unavailable for this trajectory._"

    config = load_config_for_trajectory(trajectory)
    if not config:
        return unavailable

    eval_block = config.get("eval")
    if not isinstance(eval_block, dict):
        return unavailable

    eval_types = eval_block.get("eval_types") or []
    reference_answers = eval_block.get("reference_answers")

    lines = ["## Expected Answer"]

    if eval_types:
        lines.append(f"**Evaluator(s):** {', '.join(map(str, eval_types))}")
    else:
        lines.append("**Evaluator(s):** —")

    has_reference_answers = isinstance(reference_answers, dict) and len(reference_answers) > 0
    if has_reference_answers:
        lines.append("**Reference answers:**")
        lines.append("```json")
        lines.append(json.dumps(reference_answers, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        has_reference_url = bool(eval_block.get("reference_url"))
        has_program_html = bool(eval_block.get("program_html"))
        if has_reference_url or has_program_html:
            lines.append("_Non-string evaluator — see config for full details._")
        else:
            lines.append("_Reference answer unavailable for this trajectory._")

    return "\n".join(lines)


def build_task_description(trajectory: Dict[str, Any], trajectory_id: str) -> str:
    if trajectory.get("task_description"):
        return f"## Task Description\n{trajectory['task_description']}"

    lines = [
        "## Task Description",
        f"**Trajectory ID:** `{trajectory_id}`",
        f"**Task Intent:** {trajectory.get('intent', '—')}",
        f"**Site(s):** {', '.join(map(str, trajectory.get('sites', []))) if trajectory.get('sites') else '—'}",
        f"**Start URL:** {trajectory.get('start_url', '—')}",
        "",
        "## Run Metadata",
        f"**Success:** {trajectory.get('success', '—')}",
        f"**Stop Reason:** {trajectory.get('stop_reason', '—')}",
        f"**Total Steps:** {trajectory.get('total_steps', len(get_steps(trajectory)))}",
        f"**Observation Type:** {trajectory.get('observation_type', '—')}",
        f"**Model:** {trajectory.get('model', '—')}",
        f"**Agent Variant:** {trajectory.get('agent_variant', '—')}",
        f"**Started At:** {trajectory.get('started_at', '—')}",
        f"**Ended At:** {trajectory.get('ended_at', '—')}",
        f"**Eval Score:** {trajectory.get('eval_score', '—')}",
    ]
    lines.append("")
    lines.append(build_expected_answer_md(trajectory))
    return "\n".join(lines)


def extract_dom_like_state(step: Dict[str, Any]) -> Tuple[Any, bool]:
    if "ground_truth_dom" in step and step.get("ground_truth_dom") is not None:
        return step.get("ground_truth_dom"), True
    if "dom" in step and step.get("dom") is not None:
        return step.get("dom"), True
    return None, False


def _scrollable_text_block(text: Any, max_height_px: int = 260) -> str:
    """Escaped plain text in a fixed-height scroll region (used inside Markdown with sanitize_html=False)."""
    body = html.escape(str(text) if text is not None else "")
    return (
        f'<pre style="max-height:{max_height_px}px;overflow:auto;font-size:0.82em;line-height:1.3;'
        f"margin:0.4em 0;white-space:pre-wrap;word-break:break-word;"
        f'padding:0.45em 0.55em;border-radius:6px;border:1px solid rgba(127,127,127,0.25);'
        f'background:rgba(127,127,127,0.06);">{body}</pre>'
    )


def format_step_markdown(
    step: Dict[str, Any],
    step_idx: int,
    n_steps: int,
    t_star_t: Optional[int],
) -> str:
    t_val = get_step_number(step, step_idx)
    marker = t_star_t is not None and t_val == t_star_t
    star = "\n\n**⭐ Highlighted as t\\***" if marker else ""

    url = step.get("url", "")
    action = step.get("action", step.get("agent_action", ""))
    observation = step.get("observation", step.get("agent_observation", ""))
    thought = step.get("thought", "")
    raw_prediction = step.get("raw_prediction", "")
    parse_error = step.get("parse_error", None)

    t_star_note = ""
    if t_star_t is not None:
        t_star_note = f"\n\n**t\\* (saved):** trajectory step **t = {t_star_t}**"

    blocks = [
        f"## Step t = {t_val}  (step {step_idx + 1} of {n_steps}){star}{t_star_note}",
        f"**URL:** {html.escape(url or '—')}",
        "### Agent Action",
        _scrollable_text_block(action, max_height_px=140),
        "### Agent Observation",
        _scrollable_text_block(observation, max_height_px=320),
    ]

    if thought:
        blocks.extend(["### Agent Thought", _scrollable_text_block(thought, max_height_px=200)])
    if raw_prediction:
        blocks.extend(["### Raw Prediction", _scrollable_text_block(raw_prediction, max_height_px=160)])
    if parse_error is not None:
        blocks.extend(["### Parse Error", _scrollable_text_block(parse_error, max_height_px=120)])

    return "\n\n".join(blocks)


def make_status_message(annotator_id: str, trajectory_id: str, t_star_t: Optional[int]) -> str:
    completed, total = count_completed_for_annotator(annotator_id)
    t_star_text = "not selected" if t_star_t is None else f"t = {t_star_t}"
    return (
        f"Annotator: `{annotator_id or '—'}` | "
        f"Trajectory: `{trajectory_id or '—'}` | "
        f"t*: `{t_star_text}` | "
        f"Completed: `{completed}/{total}`"
    )


def get_step_bounds(trajectory: Dict[str, Any]) -> Tuple[int, int]:
    """1-based [min, max] for the step slider (inclusive; matches 1..N in the run)."""
    steps = get_steps(trajectory)
    if not steps:
        return 1, 1
    n = len(steps)
    return 1, n


def get_step_payload(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_t: Optional[int],
    trajectory_id: str,
) -> Tuple[str, Any, bool, str, int]:
    steps = get_steps(trajectory)
    task_md = build_task_description(trajectory, trajectory_id)

    if not steps:
        return "No steps found.", None, False, task_md, 0

    step_idx = max(0, min(step_idx, len(steps) - 1))
    n = len(steps)
    step = steps[step_idx]
    step_md = format_step_markdown(step, step_idx, n, t_star_t)
    dom_state, has_dom = extract_dom_like_state(step)
    return step_md, dom_state, has_dom, task_md, step_idx


def _decode_t_star_from_annotation(ann: Optional[Dict[str, Any]], trajectory: Dict[str, Any]) -> Optional[int]:
    """Return t* as the trajectory's step t (1..N), migrating legacy 0-based list index in JSON."""
    if not ann:
        return None
    raw = ann.get("t_star_step")
    if raw is None:
        return None
    steps = get_steps(trajectory)
    n = len(steps)
    if n == 0:
        return None
    is_traj_t = ann.get("t_star_is_trajectory_t")
    raw_i = int(raw)
    if is_traj_t is True:
        for i, s in enumerate(steps):
            if get_step_number(s, i) == raw_i:
                return raw_i
        if 1 <= raw_i <= n and get_step_number(steps[raw_i - 1], raw_i - 1) == raw_i:
            return raw_i
        return None
    if 0 <= raw_i < n:
        return get_step_number(steps[raw_i], raw_i)
    return None


def build_full_view(
    annotator_id: str,
    trajectory_id: str,
    t_star_t: Optional[int],
    current_step_idx: int,
    failure_classification: Optional[str],
    notes: str,
    save_status_text: str = "",
):
    failure_update = gr.update(value=failure_classification, interactive=True)
    notes_update = gr.update(value=notes, interactive=True)
    save_btn_update = gr.update(interactive=True)

    if not trajectory_id:
        empty_status = make_status_message(annotator_id, "", t_star_t)
        return (
            {},
            0,
            t_star_t,
            empty_status,
            "No trajectory selected.",
            gr.update(value=None, visible=False),
            "## Task Description\n",
            gr.update(minimum=1, maximum=1, value=1, interactive=True),
            failure_update,
            notes_update,
            save_btn_update,
            save_status_text,
        )

    trajectory = load_trajectory(trajectory_id)
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, current_step_idx, t_star_t, trajectory_id
    )
    min_step, max_step = get_step_bounds(trajectory)
    status_md = make_status_message(annotator_id, trajectory_id, t_star_t)

    return (
        trajectory,
        resolved_step_idx,
        t_star_t,
        status_md,
        step_md,
        gr.update(value=dom_state, visible=has_dom),
        task_md,
        gr.update(
            minimum=min_step,
            maximum=max_step,
            value=resolved_step_idx + 1,
            interactive=True,
        ),
        failure_update,
        notes_update,
        save_btn_update,
        save_status_text,
    )


def init_app(annotator_id: str, trajectory_id: str):
    existing = load_annotation(annotator_id, trajectory_id)
    t_star_t: Optional[int] = None
    if existing and trajectory_id:
        try:
            traj = load_trajectory(trajectory_id)
            t_star_t = _decode_t_star_from_annotation(existing, traj)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            t_star_t = None
    failure_classification = existing.get("failure_classification") if existing else None
    notes = existing.get("notes", "") if existing else ""
    initial_step = 0
    return build_full_view(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        t_star_t=t_star_t,
        current_step_idx=initial_step,
        failure_classification=failure_classification,
        notes=notes,
    )


def on_context_change(annotator_id: str, trajectory_id: str):
    return init_app(annotator_id, trajectory_id)


def on_step_change(
    trajectory: Dict[str, Any],
    step_slider_1: float,
    t_star_t: Optional[int],
    annotator_id: str,
    trajectory_id: str,
):
    steps = get_steps(trajectory)
    n = len(steps)
    if n == 0:
        return 0, "No steps.", "No steps found.", gr.update(value=None, visible=False), build_task_description(trajectory, trajectory_id)
    u = int(round(float(step_slider_1)))
    step_idx = max(0, min(n - 1, u - 1))
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, step_idx, t_star_t, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_t)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def prev_step(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_t: Optional[int],
    annotator_id: str,
    trajectory_id: str,
):
    new_idx = max(0, step_idx - 1)
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, new_idx, t_star_t, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_t)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def next_step(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_t: Optional[int],
    annotator_id: str,
    trajectory_id: str,
):
    steps = get_steps(trajectory)
    max_idx = max(0, len(steps) - 1)
    new_idx = min(max_idx, step_idx + 1)
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, new_idx, t_star_t, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_t)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def mark_t_star(
    trajectory: Dict[str, Any],
    step_idx: int,
    annotator_id: str,
    trajectory_id: str,
):
    if not trajectory_id:
        return None, "Please select a trajectory first."
    steps = get_steps(trajectory)
    if not steps or not (0 <= step_idx < len(steps)):
        return None, "No valid step to mark."
    t_val = get_step_number(steps[step_idx], step_idx)
    status_md = make_status_message(annotator_id, trajectory_id, t_val)
    return t_val, f"Saved in memory: t* = trajectory step t = {t_val}. {status_md}"


def after_mark_t_star(
    trajectory: Dict[str, Any],
    current_step_idx: int,
    t_star_t: Optional[int],
    annotator_id: str,
    trajectory_id: str,
    failure_classification: Optional[str],
    notes: str,
    save_status: str,
):
    return build_full_view(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        t_star_t=t_star_t,
        current_step_idx=current_step_idx,
        failure_classification=failure_classification,
        notes=notes,
        save_status_text=save_status,
    )


def save_current_annotation(
    annotator_id: str,
    trajectory_id: str,
    t_star_t: Optional[int],
    failure_classification: Optional[str],
    notes: str,
):
    if not annotator_id.strip():
        return "Please enter an annotator ID before saving."
    if not trajectory_id:
        return "Please select a trajectory before saving."
    if t_star_t is None:
        return "Please mark the t* step before saving."
    if not failure_classification:
        return "Please choose either 'context-caused' or 'capability-caused' before saving."

    record = AnnotationRecord(
        annotator_id=annotator_id.strip(),
        trajectory_id=trajectory_id,
        t_star_step=int(t_star_t),
        failure_classification=failure_classification,
        notes=notes or "",
        timestamp=datetime.now(timezone.utc).isoformat(),
        t_star_is_trajectory_t=True,
    )
    out_path = save_annotation(record)
    completed, total = count_completed_for_annotator(annotator_id)
    return f"Saved annotation to `{out_path}`. Progress: {completed}/{total}."


def save_current_annotation_and_refresh(
    annotator_id: str,
    trajectory_id: str,
    t_star_t: Optional[int],
    failure_classification: Optional[str],
    notes: str,
):
    message = save_current_annotation(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        t_star_t=t_star_t,
        failure_classification=failure_classification,
        notes=notes,
    )
    dropdown_update = gr.update(choices=build_dropdown_choices(), value=trajectory_id)
    return message, dropdown_update


def refresh_dropdown_choices(trajectory_id: str):
    return gr.update(choices=build_dropdown_choices(), value=trajectory_id)


def build_app() -> gr.Blocks:
    ensure_dirs()
    initial_choices = build_dropdown_choices()
    initial_value = initial_choices[0][1] if initial_choices else None

    with gr.Blocks(title="Failed Trajectory Annotation Tool") as demo:
        gr.Markdown(
            "# Failed Trajectory Annotation Tool\n"
            "Step through the trajectory, mark **t\\***, choose failure attribution, add optional notes, then **Save**.\n"
            "\nTrajectory labels show peer-completion badges, e.g. `task_308  \u2022  [annotator_2]`, so you can coordinate overlap for inter-annotator agreement."
        )

        trajectory_state = gr.State({})
        current_step_state = gr.State(0)
        t_star_state = gr.State(None)

        with gr.Row():
            annotator_id = gr.Textbox(label="Annotator ID")
            trajectory_id = gr.Dropdown(
                label="Trajectory ID",
                choices=initial_choices,
                value=initial_value,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh peer status", scale=0)

        status_md = gr.Markdown("Select a trajectory to begin.")

        with gr.Row():
            with gr.Column(scale=3):
                task_md = gr.Markdown(
                    "## Task Description",
                    max_height=340,
                    container=True,
                )
                step_md = gr.Markdown(
                    "No trajectory loaded.",
                    sanitize_html=False,
                    max_height=560,
                    container=True,
                )
            with gr.Column(scale=2):
                dom_json = gr.JSON(label="Ground-Truth DOM State", visible=False, max_height=420)

        with gr.Row():
            prev_btn = gr.Button("Previous Step")
            next_btn = gr.Button("Next Step")
            step_slider = gr.Slider(
                label="Step (1 to N, matches trajectory t where present)",
                minimum=1,
                maximum=1,
                step=1,
                value=1,
            )
            mark_btn = gr.Button("Mark Current Step as t*")

        with gr.Column():
            gr.Markdown("## Failure Attribution")
            failure_classification = gr.Radio(
                label="Failure Attribution",
                choices=["context-caused", "capability-caused"],
                value=None,
                interactive=True,
            )
            notes = gr.Textbox(label="Optional Notes", lines=4, interactive=True)
            save_btn = gr.Button("Save Annotation", variant="primary", interactive=True)

        save_status = gr.Markdown("")

        context_outputs = [
            trajectory_state,
            current_step_state,
            t_star_state,
            status_md,
            step_md,
            dom_json,
            task_md,
            step_slider,
            failure_classification,
            notes,
            save_btn,
            save_status,
        ]

        _scroll_kw = dict(scroll_to_output=False, show_progress="minimal")

        demo.load(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id],
            outputs=context_outputs,
            **_scroll_kw,
        )

        trajectory_id.change(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id],
            outputs=context_outputs,
            **_scroll_kw,
        )

        annotator_id.change(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id],
            outputs=context_outputs,
            **_scroll_kw,
        )

        step_slider.change(
            fn=on_step_change,
            inputs=[trajectory_state, step_slider, t_star_state, annotator_id, trajectory_id],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
            **_scroll_kw,
        )

        prev_btn.click(
            fn=prev_step,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
            **_scroll_kw,
        ).then(
            fn=lambda i: i + 1,
            inputs=[current_step_state],
            outputs=[step_slider],
            scroll_to_output=False,
            show_progress="hidden",
        )

        next_btn.click(
            fn=next_step,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
            **_scroll_kw,
        ).then(
            fn=lambda i: i + 1,
            inputs=[current_step_state],
            outputs=[step_slider],
            scroll_to_output=False,
            show_progress="hidden",
        )

        mark_btn.click(
            fn=mark_t_star,
            inputs=[trajectory_state, current_step_state, annotator_id, trajectory_id],
            outputs=[t_star_state, save_status],
            **_scroll_kw,
        ).then(
            fn=after_mark_t_star,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id, failure_classification, notes, save_status],
            outputs=context_outputs,
            **_scroll_kw,
        )

        save_btn.click(
            fn=save_current_annotation_and_refresh,
            inputs=[annotator_id, trajectory_id, t_star_state, failure_classification, notes],
            outputs=[save_status, trajectory_id],
            **_scroll_kw,
        )

        refresh_btn.click(
            fn=refresh_dropdown_choices,
            inputs=[trajectory_id],
            outputs=[trajectory_id],
            scroll_to_output=False,
            show_progress="minimal",
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
