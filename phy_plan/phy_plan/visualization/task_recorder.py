"""
task_recorder.py: Record pipeline execution data for offline Rerun visualization.

Zero external dependencies — only stdlib (json, time, dataclasses).
Captures scene graph snapshots, robot poses, actions, replan events,
and LLM chain-of-thought at every key moment during task execution.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RecordFrame:
    """A single recorded event in the task execution timeline."""
    timestamp: float
    event_type: str  # "init" | "action_start" | "action_complete" | "replan" | "scene_update"

    # Spatial state
    robot_pose: Optional[List[float]] = None  # [x, y, z, qx, qy, qz, qw]
    scene_graph: Optional[Dict[str, Any]] = None

    # Action info
    action: Optional[Dict[str, Any]] = None
    action_index: Optional[int] = None
    action_success: Optional[bool] = None
    action_error: Optional[str] = None

    # Plan state
    plan: Optional[List[Dict[str, Any]]] = None
    highlight_ids: Optional[List[str]] = None

    # LLM / reasoning
    chain_of_thought: Optional[str] = None
    replan_context: Optional[str] = None
    replan_reason: Optional[str] = None
    physics_validation: Optional[Dict[str, Any]] = None

    # Dynamic object tracking (objects held by robot, updated per-frame via OG API)
    held_objects: Optional[List[Dict[str, Any]]] = None  # [{"name": str, "pose": [x,y,z,qx,qy,qz,qw]}]

    # Extra
    extra: Optional[Dict[str, Any]] = None


class TaskRecorder:
    """
    Lightweight recorder that accumulates ``RecordFrame`` objects and
    serialises them to a single JSON file.

    Intended to run inside the Isaac Sim / OmniGibson process.
    No dependency on ``rerun-sdk`` or ``numpy``.
    """

    def __init__(self, output_dir: str = "exp_results"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: List[RecordFrame] = []
        self._instruction: str = ""
        self._highlight_ids: List[str] = []
        self._recording_active: bool = False

    # ------------------------------------------------------------------ #
    #  Public recording API                                                #
    # ------------------------------------------------------------------ #

    def start_recording(self, instruction: str, highlight_ids: Optional[List[str]] = None):
        self._frames.clear()
        self._instruction = instruction
        self._highlight_ids = highlight_ids or []
        self._recording_active = True

    def stop_recording(self):
        self._recording_active = False

    @property
    def is_recording(self) -> bool:
        return self._recording_active

    # -- Event helpers -------------------------------------------------- #

    def record_init(
        self,
        scene_graph_dict: Optional[Dict] = None,
        robot_pose: Optional[List[float]] = None,
        plan: Optional[List[Dict]] = None,
        chain_of_thought: Optional[str] = None,
        held_objects: Optional[List[Dict]] = None,
    ):
        self._append(RecordFrame(
            timestamp=time.time(),
            event_type="init",
            robot_pose=robot_pose,
            scene_graph=scene_graph_dict,
            plan=plan,
            highlight_ids=self._highlight_ids,
            chain_of_thought=chain_of_thought,
            held_objects=held_objects,
        ))

    def record_action_start(
        self,
        action_dict: Dict,
        action_index: int,
        scene_graph_dict: Optional[Dict] = None,
        robot_pose: Optional[List[float]] = None,
    ):
        self._append(RecordFrame(
            timestamp=time.time(),
            event_type="action_start",
            robot_pose=robot_pose,
            scene_graph=scene_graph_dict,
            action=action_dict,
            action_index=action_index,
            highlight_ids=self._highlight_ids,
        ))

    def record_action_complete(
        self,
        action_dict: Dict,
        action_index: int,
        success: bool,
        error: Optional[str] = None,
        scene_graph_dict: Optional[Dict] = None,
        robot_pose: Optional[List[float]] = None,
        held_objects: Optional[List[Dict]] = None,
    ):
        self._append(RecordFrame(
            timestamp=time.time(),
            event_type="action_complete",
            robot_pose=robot_pose,
            scene_graph=scene_graph_dict,
            action=action_dict,
            action_index=action_index,
            action_success=success,
            action_error=error,
            highlight_ids=self._highlight_ids,
            held_objects=held_objects,
        ))

    def record_replan(
        self,
        reason: str,
        context: Optional[str] = None,
        new_plan: Optional[List[Dict]] = None,
        scene_graph_dict: Optional[Dict] = None,
        robot_pose: Optional[List[float]] = None,
        chain_of_thought: Optional[str] = None,
        physics_validation: Optional[Dict] = None,
        held_objects: Optional[List[Dict]] = None,
    ):
        self._append(RecordFrame(
            timestamp=time.time(),
            event_type="replan",
            robot_pose=robot_pose,
            scene_graph=scene_graph_dict,
            plan=new_plan,
            replan_reason=reason,
            replan_context=context,
            chain_of_thought=chain_of_thought,
            physics_validation=physics_validation,
            highlight_ids=self._highlight_ids,
            held_objects=held_objects,
        ))

    def record_scene_update(
        self,
        scene_graph_dict: Optional[Dict] = None,
        robot_pose: Optional[List[float]] = None,
        held_objects: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None,
    ):
        self._append(RecordFrame(
            timestamp=time.time(),
            event_type="scene_update",
            robot_pose=robot_pose,
            scene_graph=scene_graph_dict,
            held_objects=held_objects,
            highlight_ids=self._highlight_ids,
            extra=extra,
        ))

    # -- Persistence ---------------------------------------------------- #

    def save(self, filename: Optional[str] = None) -> str:
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{ts}.json"

        path = self._output_dir / filename
        data = {
            "instruction": self._instruction,
            "highlight_ids": self._highlight_ids,
            "frame_count": len(self._frames),
            "frames": [self._frame_to_dict(f) for f in self._frames],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

        print(f"[TaskRecorder] Saved {len(self._frames)} frames → {path}")
        return str(path)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _append(self, frame: RecordFrame):
        if self._recording_active:
            self._frames.append(frame)

    @staticmethod
    def _frame_to_dict(frame: RecordFrame) -> Dict[str, Any]:
        d = asdict(frame)
        return {k: v for k, v in d.items() if v is not None}
