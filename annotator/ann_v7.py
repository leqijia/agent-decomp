from __future__ import annotations

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


def get_step_number(step: Dict[str, Any], fallback_idx: int) -> int:
    if step.get("step_id") is not None:
        return int(step["step_id"])
    if step.get("t") is not None:
        return int(step["t"])
    return fallback_idx


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


def format_step_markdown(step: Dict[str, Any], step_idx: int, t_star_step: Optional[int], mode: str) -> str:
    step_num = get_step_number(step, step_idx)
    marker = "\n\n**⭐ Highlighted as t\\***" if t_star_step == step_idx else ""

    url = step.get("url", "")
    action = step.get("action", step.get("agent_action", ""))
    observation = step.get("observation", step.get("agent_observation", ""))
    thought = step.get("thought", "")
    raw_prediction = step.get("raw_prediction", "")
    parse_error = step.get("parse_error", None)

    blocks = [
        f"## Step {step_idx}  \n**Step Number:** {step_num}{marker}",
        f"**URL:** {url or '—'}",
    ]

    if mode == "failure attribution" and t_star_step is not None:
        blocks.append(f"**Review Mode:** failure attribution  \n**Locked t\\* step index:** `{t_star_step}`")

    blocks.extend([
        "### Agent Action",
        f"```\n{action}\n```",
        "### Agent Observation",
        f"```\n{observation}\n```",
    ])

    if thought:
        blocks.extend(["### Agent Thought", f"```\n{thought}\n```"])
    if raw_prediction:
        blocks.extend(["### Raw Prediction", f"```\n{raw_prediction}\n```"])
    if parse_error is not None:
        blocks.extend(["### Parse Error", f"```\n{parse_error}\n```"])

    return "\n\n".join(blocks)


def make_status_message(annotator_id: str, trajectory_id: str, t_star_step: Optional[int], mode: str) -> str:
    completed, total = count_completed_for_annotator(annotator_id)
    t_star_text = "not selected" if t_star_step is None else str(t_star_step)
    return (
        f"Annotator: `{annotator_id or '—'}` | "
        f"Trajectory: `{trajectory_id or '—'}` | "
        f"Mode: `{mode}` | "
        f"t*: `{t_star_text}` | "
        f"Completed: `{completed}/{total}`"
    )


def get_step_bounds(trajectory: Dict[str, Any]) -> Tuple[int, int]:
    steps = get_steps(trajectory)
    if not steps:
        return 0, 0
    return 0, max(0, len(steps) - 1)


def get_step_payload(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_step: Optional[int],
    mode: str,
    trajectory_id: str,
) -> Tuple[str, Any, bool, str, int]:
    steps = get_steps(trajectory)
    task_md = build_task_description(trajectory, trajectory_id)

    if not steps:
        return "No steps found.", None, False, task_md, 0

    if mode == "failure attribution" and t_star_step is not None:
        step_idx = t_star_step

    step_idx = max(0, min(step_idx, len(steps) - 1))
    step = steps[step_idx]
    step_md = format_step_markdown(step, step_idx, t_star_step, mode)
    dom_state, has_dom = extract_dom_like_state(step)
    return step_md, dom_state, has_dom, task_md, step_idx


def compute_effective_mode(requested_mode: str, annotator_id: str, t_star_step: Optional[int]) -> Tuple[str, str]:
    annotator_ok = bool(annotator_id.strip())
    has_t_star = t_star_step is not None
    can_enter_failure = annotator_ok and has_t_star

    if requested_mode == "failure attribution" and not can_enter_failure:
        if not annotator_ok and not has_t_star:
            return "t* labeling", "Enter Annotator ID and mark t* before switching to failure attribution."
        if not annotator_ok:
            return "t* labeling", "Enter Annotator ID before switching to failure attribution."
        return "t* labeling", "Mark t* before switching to failure attribution."

    if requested_mode == "failure attribution":
        return "failure attribution", f"**Failure attribution mode:** reviewing locked t* step `{t_star_step}`."

    if not annotator_ok and not has_t_star:
        return "t* labeling", "Enter Annotator ID and mark t* to unlock failure attribution."
    if not annotator_ok:
        return "t* labeling", "Enter Annotator ID to unlock failure attribution."
    if not has_t_star:
        return "t* labeling", "Mark t* to unlock failure attribution."
    return "t* labeling", "Failure attribution is unlocked."


def build_mode_update(effective_mode: str, annotator_id: str, t_star_step: Optional[int]):
    annotator_ok = bool(annotator_id.strip())
    has_t_star = t_star_step is not None
    choices = ["t* labeling"]
    if annotator_ok and has_t_star:
        choices.append("failure attribution")
    return gr.update(choices=choices, value=effective_mode)


def build_full_view(
    annotator_id: str,
    trajectory_id: str,
    requested_mode: str,
    t_star_step: Optional[int],
    current_step_idx: int,
    failure_classification: Optional[str],
    notes: str,
    save_status_text: str = "",
):
    effective_mode, helper_text = compute_effective_mode(requested_mode, annotator_id, t_star_step)
    mode_update = build_mode_update(effective_mode, annotator_id, t_star_step)

    t_mode = effective_mode == "t* labeling"
    a_mode = effective_mode == "failure attribution"

    mark_btn_update = gr.update(visible=t_mode)
    prev_update = gr.update(interactive=t_mode)
    next_update = gr.update(interactive=t_mode)
    failure_update = gr.update(value=failure_classification, interactive=a_mode)
    notes_update = gr.update(value=notes, interactive=a_mode)
    save_btn_update = gr.update(interactive=a_mode)
    mode_warning_update = gr.update(value=helper_text, visible=bool(helper_text))

    if not trajectory_id:
        empty_status = make_status_message(annotator_id, "", t_star_step, effective_mode)
        return (
            mode_update,
            {},
            0,
            t_star_step,
            empty_status,
            "No trajectory selected.",
            gr.update(value=None, visible=False),
            "## Task Description\n",
            gr.update(minimum=0, maximum=0, value=0, interactive=t_mode),
            failure_update,
            notes_update,
            mark_btn_update,
            prev_update,
            next_update,
            mode_warning_update,
            save_btn_update,
            save_status_text,
        )

    trajectory = load_trajectory(trajectory_id)
    display_step = t_star_step if a_mode and t_star_step is not None else current_step_idx
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, display_step, t_star_step, effective_mode, trajectory_id
    )
    min_step, max_step = get_step_bounds(trajectory)
    status_md = make_status_message(annotator_id, trajectory_id, t_star_step, effective_mode)

    return (
        mode_update,
        trajectory,
        resolved_step_idx,
        t_star_step,
        status_md,
        step_md,
        gr.update(value=dom_state, visible=has_dom),
        task_md,
        gr.update(minimum=min_step, maximum=max_step, value=resolved_step_idx, interactive=t_mode),
        failure_update,
        notes_update,
        mark_btn_update,
        prev_update,
        next_update,
        mode_warning_update,
        save_btn_update,
        save_status_text,
    )


def init_app(annotator_id: str, trajectory_id: str, mode: str):
    existing = load_annotation(annotator_id, trajectory_id)
    t_star_step = existing.get("t_star_step") if existing else None
    failure_classification = existing.get("failure_classification") if existing else None
    notes = existing.get("notes", "") if existing else ""
    initial_step = t_star_step if (mode == "failure attribution" and t_star_step is not None) else 0
    return build_full_view(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        requested_mode=mode,
        t_star_step=t_star_step,
        current_step_idx=initial_step,
        failure_classification=failure_classification,
        notes=notes,
    )


def on_context_change(annotator_id: str, trajectory_id: str, mode: str):
    return init_app(annotator_id, trajectory_id, mode)


def on_mode_change(
    annotator_id: str,
    trajectory_id: str,
    requested_mode: str,
    t_star_step: Optional[int],
    current_step_idx: int,
    failure_classification: Optional[str],
    notes: str,
    save_status: str,
):
    return build_full_view(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        requested_mode=requested_mode,
        t_star_step=t_star_step,
        current_step_idx=current_step_idx,
        failure_classification=failure_classification,
        notes=notes,
        save_status_text=save_status,
    )


def on_mode_change_view_only(
    annotator_id: str,
    trajectory_id: str,
    requested_mode: str,
    t_star_step: Optional[int],
    current_step_idx: int,
    failure_classification: Optional[str],
    notes: str,
    save_status: str,
):
    full = on_mode_change(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        requested_mode=requested_mode,
        t_star_step=t_star_step,
        current_step_idx=current_step_idx,
        failure_classification=failure_classification,
        notes=notes,
        save_status=save_status,
    )
    return full[1:]


def on_step_change(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_step: Optional[int],
    annotator_id: str,
    trajectory_id: str,
    mode: str,
):
    effective_mode, _ = compute_effective_mode(mode, annotator_id, t_star_step)
    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, step_idx, t_star_step, effective_mode, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_step, effective_mode)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def prev_step(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_step: Optional[int],
    annotator_id: str,
    trajectory_id: str,
    mode: str,
):
    effective_mode, _ = compute_effective_mode(mode, annotator_id, t_star_step)
    if effective_mode == "failure attribution" and t_star_step is not None:
        new_idx = t_star_step
    else:
        new_idx = max(0, step_idx - 1)

    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, new_idx, t_star_step, effective_mode, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_step, effective_mode)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def next_step(
    trajectory: Dict[str, Any],
    step_idx: int,
    t_star_step: Optional[int],
    annotator_id: str,
    trajectory_id: str,
    mode: str,
):
    effective_mode, _ = compute_effective_mode(mode, annotator_id, t_star_step)
    if effective_mode == "failure attribution" and t_star_step is not None:
        new_idx = t_star_step
    else:
        steps = get_steps(trajectory)
        max_idx = max(0, len(steps) - 1)
        new_idx = min(max_idx, step_idx + 1)

    step_md, dom_state, has_dom, task_md, resolved_step_idx = get_step_payload(
        trajectory, new_idx, t_star_step, effective_mode, trajectory_id
    )
    status_md = make_status_message(annotator_id, trajectory_id, t_star_step, effective_mode)
    return resolved_step_idx, status_md, step_md, gr.update(value=dom_state, visible=has_dom), task_md


def mark_t_star(
    trajectory: Dict[str, Any],
    step_idx: int,
    annotator_id: str,
    trajectory_id: str,
    mode: str,
):
    effective_mode, _ = compute_effective_mode(mode, annotator_id, None)
    if effective_mode != "t* labeling":
        return None, "Switch to t* labeling mode to choose the first unrecoverable mistake."
    if not trajectory_id:
        return None, "Please select a trajectory first."
    status_md = make_status_message(annotator_id, trajectory_id, step_idx, "t* labeling")
    return step_idx, f"Saved in memory: t* set to step {step_idx}. {status_md}"


def after_mark_t_star(
    trajectory: Dict[str, Any],
    current_step_idx: int,
    t_star_step: Optional[int],
    annotator_id: str,
    trajectory_id: str,
    mode: str,
    failure_classification: Optional[str],
    notes: str,
    save_status: str,
):
    return build_full_view(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        requested_mode=mode,
        t_star_step=t_star_step,
        current_step_idx=current_step_idx,
        failure_classification=failure_classification,
        notes=notes,
        save_status_text=save_status,
    )


def save_current_annotation(
    annotator_id: str,
    trajectory_id: str,
    t_star_step: Optional[int],
    failure_classification: Optional[str],
    notes: str,
    mode: str,
):
    effective_mode, _ = compute_effective_mode(mode, annotator_id, t_star_step)
    if effective_mode != "failure attribution":
        return "Saving is only allowed in failure attribution mode."
    if not annotator_id.strip():
        return "Please enter an annotator ID before saving."
    if not trajectory_id:
        return "Please select a trajectory before saving."
    if t_star_step is None:
        return "Please mark the t* step before entering failure attribution mode."
    if not failure_classification:
        return "Please choose either 'context-caused' or 'capability-caused' before saving."

    record = AnnotationRecord(
        annotator_id=annotator_id.strip(),
        trajectory_id=trajectory_id,
        t_star_step=int(t_star_step),
        failure_classification=failure_classification,
        notes=notes or "",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    out_path = save_annotation(record)
    completed, total = count_completed_for_annotator(annotator_id)
    return f"Saved annotation to `{out_path}`. Progress: {completed}/{total}."


def save_current_annotation_and_refresh(
    annotator_id: str,
    trajectory_id: str,
    t_star_step: Optional[int],
    failure_classification: Optional[str],
    notes: str,
    mode: str,
):
    message = save_current_annotation(
        annotator_id=annotator_id,
        trajectory_id=trajectory_id,
        t_star_step=t_star_step,
        failure_classification=failure_classification,
        notes=notes,
        mode=mode,
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
            "1. **t* labeling**: inspect the failed trajectory step by step and mark the first unrecoverable mistake.\n"
            "2. **failure attribution**: review the locked t* step and save a classification.\n"
            "\nTrajectory labels show peer-completion badges, e.g. `task_308  \u2022  [Muhammad]`, so you can coordinate overlap for inter-annotator agreement."
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
            mode = gr.Radio(
                label="Annotation Mode",
                choices=["t* labeling"],
                value="t* labeling",
                interactive=True,
            )

        status_md = gr.Markdown("Select a trajectory to begin.")
        mode_warning = gr.Markdown(visible=False)

        with gr.Row():
            with gr.Column(scale=3):
                task_md = gr.Markdown("## Task Description")
                step_md = gr.Markdown("No trajectory loaded.")
            with gr.Column(scale=2):
                dom_json = gr.JSON(label="Ground-Truth DOM State", visible=False)

        with gr.Row():
            prev_btn = gr.Button("Previous Step")
            next_btn = gr.Button("Next Step")
            step_slider = gr.Slider(label="Step", minimum=0, maximum=0, step=1, value=0)
            mark_btn = gr.Button("Mark Current Step as t*")

        with gr.Column():
            gr.Markdown("## Failure Attribution")
            failure_classification = gr.Radio(
                label="Failure Attribution",
                choices=["context-caused", "capability-caused"],
                value=None,
                interactive=False,
            )
            notes = gr.Textbox(label="Optional Notes", lines=4, interactive=False)
            save_btn = gr.Button("Save Annotation", variant="primary", interactive=False)

        save_status = gr.Markdown("")

        context_outputs = [
            mode,
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
            mark_btn,
            prev_btn,
            next_btn,
            mode_warning,
            save_btn,
            save_status,
        ]
        mode_switch_outputs = context_outputs[1:]

        demo.load(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id, mode],
            outputs=context_outputs,
        )

        trajectory_id.change(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id, mode],
            outputs=context_outputs,
        )

        annotator_id.change(
            fn=on_context_change,
            inputs=[annotator_id, trajectory_id, mode],
            outputs=context_outputs,
        )

        mode.change(
            fn=on_mode_change_view_only,
            inputs=[annotator_id, trajectory_id, mode, t_star_state, current_step_state, failure_classification, notes, save_status],
            outputs=mode_switch_outputs,
        )

        step_slider.change(
            fn=on_step_change,
            inputs=[trajectory_state, step_slider, t_star_state, annotator_id, trajectory_id, mode],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
        )

        prev_btn.click(
            fn=prev_step,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id, mode],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
        ).then(
            fn=lambda x: x,
            inputs=[current_step_state],
            outputs=[step_slider],
        )

        next_btn.click(
            fn=next_step,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id, mode],
            outputs=[current_step_state, status_md, step_md, dom_json, task_md],
        ).then(
            fn=lambda x: x,
            inputs=[current_step_state],
            outputs=[step_slider],
        )

        mark_btn.click(
            fn=mark_t_star,
            inputs=[trajectory_state, current_step_state, annotator_id, trajectory_id, mode],
            outputs=[t_star_state, save_status],
        ).then(
            fn=after_mark_t_star,
            inputs=[trajectory_state, current_step_state, t_star_state, annotator_id, trajectory_id, mode, failure_classification, notes, save_status],
            outputs=context_outputs,
        )

        save_btn.click(
            fn=save_current_annotation_and_refresh,
            inputs=[annotator_id, trajectory_id, t_star_state, failure_classification, notes, mode],
            outputs=[save_status, trajectory_id],
        )

        refresh_btn.click(
            fn=refresh_dropdown_choices,
            inputs=[trajectory_id],
            outputs=[trajectory_id],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
