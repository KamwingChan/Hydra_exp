"""
Primitive action controller with task queue and progress monitoring.

Supports multi-step task sequences (e.g., navigate → grasp → navigate → place)
with automatic step advancement.
"""
import rospy
import time
import threading
import torch
import gc
import omnigibson as og
from omnigibson.utils.ui_utils import KeyboardEventHandler
import omnigibson.lazy as lazy
from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class PrimitiveStep:
    """A single step in a primitive task sequence.
    
    Attributes:
        primitive_type: Type of primitive action. One of:
            "navigate_to", "grasp", "place_on_top", "place_inside",
            "open", "close", "release", "toggle_on", "toggle_off", "observe"
        target_name: Name/keyword to search for in scene objects.
        node_id: Original scene graph node ID (e.g. "O(13)") for position-based disambiguation.
    """
    primitive_type: str
    target_name: str
    node_id: str = ""


class PrimitiveController:
    """Handles semantic action primitives with task queue and progress monitoring.
    
    Supports executing a sequence of primitive steps automatically.
    Each step completes before the next one begins.
    """
    
    def __init__(self, env, robot, curobo_batch_size=1, execution_mode=None):
        """
        Initialize primitive controller.
        
        Args:
            env: OmniGibson environment
            robot: Robot instance
            curobo_batch_size: Batch size for CuRobo (default 1 for 8GB GPU)
            execution_mode: ExecutionMode.FULL or ExecutionMode.SYMBOLIC (default: FULL)
        """
        self.env = env
        self.robot = robot
        self.curobo_batch_size = curobo_batch_size
        self.execution_mode = execution_mode  # 保存执行模式
        
        self._primitive_api = None
        self._primitive_controller = None  # generator for current primitive action
        self._g_key_pressed = False  # G 键按下标志
        self._p_key_pressed = False  # P 键按下标志 (trigger PhyPlanPipeline)
        self._pipeline = None  # PhyPlanPipeline reference
        self._pipeline_factory: Optional[Callable] = None  # lazy init factory
        self._mode_switch_callback: Optional[Callable] = None  # auto-switch mode
        
        # 任务队列
        self._task_queue: List[PrimitiveStep] = []
        self._current_step_index: int = 0
        self._task_running: bool = False
        
        # 回调：供 PhyPlanPipeline 等外部编排器监听执行进度
        # on_step_complete(step_index: int, step: PrimitiveStep, success: bool, error: str)
        self._on_step_complete: Optional[Callable] = None
        # on_task_complete(success: bool, error: str)
        self._on_task_complete: Optional[Callable] = None
        
        # 重规划支持：回调中安全替换任务队列（避免重入问题）
        self._replan_steps: Optional[List[PrimitiveStep]] = None
        
        # 注册 G/P/V 键回调
        try:
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.G,
                self._on_g_key_pressed
            )
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.P,
                self._on_p_key_pressed
            )
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.V,
                self._on_v_key_pressed
            )
            rospy.loginfo("Registered G/P/V key callbacks for primitive control")
        except Exception as e:
            rospy.logwarn(f"Failed to register keyboard callbacks: {e}")
    
    def _on_g_key_pressed(self):
        """G 键按下时的回调函数"""
        control_mode = getattr(self, '_current_mode', 'primitive')
        rospy.loginfo(f"G key callback triggered, control_mode={control_mode}, task_running={self._task_running}")
        if control_mode == "primitive" and not self._task_running:
            self._g_key_pressed = True
            rospy.loginfo("G key pressed - will start task sequence in next frame")
        else:
            if control_mode != "primitive":
                rospy.logwarn(f"G key pressed but control_mode is '{control_mode}', not 'primitive'. Switch to primitive mode first!")
            if self._task_running:
                rospy.logwarn("G key pressed but task sequence is already running")
    
    def _on_p_key_pressed(self):
        """P 键按下时的回调函数 - 触发 PhyPlanPipeline
        
        行为：
        1. 自动切换到 primitive 模式（如果当前不是）
        2. 首次按下时 lazy init pipeline（通过 factory）
        3. 设置标志，下一帧触发交互式规划
        """
        if self._task_running:
            rospy.logwarn("P key pressed but task sequence is already running")
            return
        
        # 自动切换到 primitive 模式
        control_mode = getattr(self, '_current_mode', 'primitive')
        if control_mode != "primitive":
            rospy.loginfo(f"[P key] Auto-switching from '{control_mode}' to 'primitive' mode")
            self._current_mode = "primitive"
            if self._mode_switch_callback:
                self._mode_switch_callback("primitive")
        
        # Lazy init: 首次按 P 时通过 factory 创建 pipeline
        if self._pipeline is None:
            if self._pipeline_factory is not None:
                rospy.loginfo("[P key] First press - initializing pipeline...")
                try:
                    self._pipeline = self._pipeline_factory()
                    if self._pipeline is not None:
                        rospy.loginfo("[P key] Pipeline initialized successfully")
                    else:
                        rospy.logerr("[P key] Pipeline factory returned None")
                        return
                except Exception as e:
                    rospy.logerr(f"[P key] Pipeline initialization failed: {e}")
                    return
            else:
                rospy.logwarn("P key pressed but no pipeline or factory configured. "
                              "Use G key for hardcoded experiment.")
                return
        
        self._p_key_pressed = True
        rospy.loginfo("P key pressed - will trigger interactive planning in next frame")
    
    def _on_v_key_pressed(self):
        """V 键按下时的回调函数 - 取消当前任务 / 销毁 pipeline
        
        行为：
        - 如果有任务在执行：取消当前任务（pipeline 保留，可再次 P 键规划）
        - 如果没有任务但 pipeline 已初始化：销毁 pipeline，释放资源
        - 如果都没有：不做任何操作
        """
        if self._task_running:
            rospy.loginfo("[V key] Cancelling current task execution...")
            if self._pipeline is not None:
                self._pipeline.cancel()
            else:
                self.cancel_task()
        elif self._pipeline is not None:
            rospy.loginfo("[V key] Destroying pipeline...")
            self._pipeline.destroy()
            self._pipeline = None
            rospy.loginfo("[V key] Pipeline destroyed. P key will re-initialize.")
        else:
            rospy.loginfo("[V key] Nothing to cancel or destroy")
    
    def set_pipeline(self, pipeline):
        """Set PhyPlanPipeline reference directly."""
        self._pipeline = pipeline
    
    def set_pipeline_factory(self, factory: Callable, mode_switch_callback: Callable = None):
        """
        Set a factory callable for lazy pipeline initialization.
        
        The factory is called on first P key press. It should return a
        PhyPlanPipeline instance (or None on failure).
        
        Args:
            factory: Callable that returns PhyPlanPipeline
            mode_switch_callback: Optional callback to switch control mode,
                                  called with mode string (e.g., "primitive")
        """
        self._pipeline_factory = factory
        self._mode_switch_callback = mode_switch_callback
        rospy.loginfo("[PrimitiveController] Pipeline factory registered "
                      "(P key will lazy-init on first press)")
    
    def request_replan(self, steps: List[PrimitiveStep]):
        """
        Request task queue replacement during step completion callback.
        
        Safe to call from _on_step_complete callback — the replacement
        is applied after the callback returns, inside _advance_to_next_step()
        or get_action() exception handler. Avoids re-entrancy issues.
        
        Args:
            steps: New PrimitiveStep list to replace the current queue
        """
        self._replan_steps = steps
        rospy.loginfo(f"[TaskQueue] Replan requested: {len(steps)} new steps queued")
    
    def _start_hardcoded_experiment(self):
        """Start the hardcoded experiment task sequence (fallback for G key)."""
        experiment_steps = [
            PrimitiveStep("navigate_to", "coffee"),
            PrimitiveStep("grasp", "coffee"),
            PrimitiveStep("navigate_to", "conference_table"),
            PrimitiveStep("place_on_top", "conference_table"),
        ]
        self.start_task_sequence(experiment_steps)
    
    def cancel_task(self):
        """
        Cancel the currently running task and return to idle.
        
        Clears the task queue, resets state, releases the primitive controller.
        Safe to call at any time.
        """
        if not self._task_running:
            rospy.loginfo("[TaskQueue] No task to cancel")
            return
        
        rospy.loginfo("[TaskQueue] Cancelling current task...")
        self._task_queue = []
        self._current_step_index = 0
        self._task_running = False
        self._primitive_controller = None
        self._replan_steps = None
        self._on_step_complete = None
        self._on_task_complete = None
        rospy.loginfo("[TaskQueue] Task cancelled, returning to idle")
    
    def set_mode(self, mode):
        """Set current control mode (for G/P key callback)."""
        self._current_mode = mode
    
    # ==================== Primitive API Initialization ====================
    
    def _initialize_primitive_api(self):
        """Lazy initialization of BehaviorActionAPI with progress feedback."""
        if self._primitive_api is not None:
            return True
        
        try:
            import time
            import threading
            
            # init GPU memory
            gc.collect()  # Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                rospy.loginfo(f"GPU memory before curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                rospy.loginfo(f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
            
            from phy_plan.executor import BehaviorActionAPI, ExecutionMode
            
            # if SYMBOLIC mode is not specified, use SYMBOLIC mode
            if self.execution_mode is None:
                execution_mode = ExecutionMode.SYMBOLIC
            else:
                execution_mode = self.execution_mode
            
            # progress feedback (only show in FULL mode)
            progress_info = {'active': True, 'start_time': time.time()}
            
            def show_init_progress():
                """background thread to show initialization progress"""
                stages = [
                    (5, "Loading robot configuration..."),
                    (15, "Building collision checker..."),
                    (30, "Initializing motion planners (IK/TrajOpt)..."),
                    (60, "Warming up CUDA kernels (this may take 1-2 minutes)..."),
                ]
                stage_idx = 0
                
                while progress_info['active']:
                    elapsed = time.time() - progress_info['start_time']
                    
                    # update stage prompt
                    if stage_idx < len(stages) - 1 and elapsed > stages[stage_idx + 1][0]:
                        stage_idx += 1
                    
                    if elapsed > 3.0:  # show after 3 seconds
                        stage_msg = stages[min(stage_idx, len(stages) - 1)][1]
                        rospy.loginfo(f"[CuRobo Init] {stage_msg} (elapsed: {elapsed:.1f}s)")
                    
                    time.sleep(5.0)  # update every 5 seconds
            
            # only start progress thread in FULL mode
            if execution_mode == ExecutionMode.FULL:
                progress_thread = threading.Thread(target=show_init_progress, daemon=True)
                progress_thread.start()
            
            try:
                # show different prompts based on mode
                if execution_mode == ExecutionMode.SYMBOLIC:
                    rospy.loginfo("Initializing BehaviorActionAPI in SYMBOLIC mode (no CuRobo, teleport + physics)...")
                    rospy.loginfo("This will be fast and use minimal GPU memory")
                    motion_cfg_kwargs = None
                else:
                    # use batch_size=1 to reduce GPU memory usage (default is 3, which may not be enough for 8GB GPU)
                    rospy.loginfo("Initializing BehaviorActionAPI with curobo_batch_size=1...")
                    rospy.loginfo("This may take 1-3 minutes. Please wait...")
                    
                    # optimized motion_cfg_kwargs configuration to reduce GPU memory usage and initialization time
                    motion_cfg_kwargs = {
                        'num_ik_seeds': 64,
                        'num_batch_ik_seeds': 64,
                        'num_trajopt_seeds': 2,
                        'num_graph_seeds': 2,
                        'ik_opt_iters': 50,
                        'finetune_trajopt_iters': 50,
                        'trajopt_tsteps': 16,
                    }
                    rospy.loginfo("Using optimized motion_cfg_kwargs to reduce GPU memory usage")
                
                self._primitive_api = BehaviorActionAPI(
                    self.env, 
                    self.robot,
                    mode=execution_mode,
                    curobo_batch_size=self.curobo_batch_size,
                    motion_cfg_kwargs=motion_cfg_kwargs
                )
                
                # stop progress feedback
                if execution_mode == ExecutionMode.FULL:
                    progress_info['active'] = False
                elapsed = time.time() - progress_info['start_time']
                
                mode_str = "SYMBOLIC" if execution_mode == ExecutionMode.SYMBOLIC else "FULL"
                rospy.loginfo(f"BehaviorActionAPI initialized successfully in {elapsed:.1f}s (mode: {mode_str})")
                if torch.cuda.is_available() and execution_mode == ExecutionMode.FULL:
                    rospy.loginfo(f"GPU memory after curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                return True
            except Exception as e:
                progress_info['active'] = False
                raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                rospy.logerr(f"GPU out of memory when initializing curobo. Current GPU usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                rospy.logerr("Suggestions: 1. Close other GPU apps  2. Reduce scene complexity  3. Restart simulation")
                import traceback
                rospy.logerr(traceback.format_exc())
            else:
                rospy.logerr(f"Failed to init BehaviorActionAPI: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
            self._primitive_api = None
            return False
        except Exception as e:
            rospy.logerr(f"Failed to init BehaviorActionAPI: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self._primitive_api = None
            return False
    
    # ==================== Object Finding ====================
    
    def _find_target_object(self, target_name="coffee", node_id=""):
        """
        Find target object in scene by name or category keyword.
        
        When multiple candidates match, uses scene_graph position data (via node_id)
        to select the closest one.
        
        Args:
            target_name: Name or category keyword to search for
            node_id: Original scene graph node ID (e.g. "O(13)") for disambiguation
            
        Returns:
            target: Object instance or None
        """
        candidates = []

        # Match by name first, then by category — collect all matches
        for obj in self.env.scene.objects:
            if hasattr(obj, 'name') and target_name.lower() in obj.name.lower():
                candidates.append(obj)

        if not candidates:
            for obj in self.env.scene.objects:
                if hasattr(obj, 'category') and target_name.lower() in str(obj.category).lower():
                    candidates.append(obj)

        if not candidates:
            rospy.logwarn(f"Target '{target_name}' not found. Available objects:")
            for obj in self.env.scene.objects:
                if hasattr(obj, 'name'):
                    rospy.logwarn(f"  - {obj.name} (category: {getattr(obj, 'category', 'N/A')})")
            return None

        if len(candidates) == 1:
            rospy.loginfo(f"Found target: {candidates[0].name}")
            return candidates[0]

        # Multiple candidates — use scene_graph position to pick the closest one
        scene_graph = getattr(self, '_scene_graph', None)
        if node_id and scene_graph:
            node = scene_graph.get_object(node_id)
            if node and node.position:
                import numpy as np
                target_pos = np.array(node.position)
                best = min(candidates, key=lambda o: np.linalg.norm(
                    np.array(o.get_position()) - target_pos
                ))
                rospy.loginfo(
                    f"Multiple matches for '{target_name}', selected closest to {node_id}: {best.name}"
                )
                return best

        rospy.logwarn(
            f"Multiple matches for '{target_name}' and no position info; using first: {candidates[0].name}"
        )
        return candidates[0]
    
    def _create_observe_generator(self, target, idle_frames=45):
        """
        Generator for observe: idle for N frames to let phy_graph update.
        
        Navigation to the object should already be done by a preceding navigate_to step.
        Idling gives the perception system time to update physical property estimates.
        
        Args:
            target: OmniGibson object to observe (unused here, kept for logging)
            idle_frames: How many frames to stay still (default 45 ≈ 3s @ 15Hz)
        """
        rospy.loginfo(f"Observing {target.name} for {idle_frames} frames...")
        for _ in range(idle_frames):
            yield self._get_idle_action()

    def _get_idle_action(self):
        """Return a zero-velocity action for the robot to stay still."""
        import numpy as np
        action_space = self.robot.action_space
        return np.zeros(action_space.shape)

    # ==================== Primitive Type Mapping ====================
    
    def _get_primitive_enum(self, primitive_type: str):
        """
        Map primitive_type string to the corresponding PrimitiveSet enum value.
        
        Args:
            primitive_type: String name like "navigate_to", "grasp", "place_on_top"
            
        Returns:
            Enum value from the active PrimitiveSet
            
        Raises:
            ValueError: If primitive_type is not recognized
        """
        primitive_set = self._primitive_api._primitive_set
        
        mapping = {
            "navigate_to": primitive_set.NAVIGATE_TO,
            "grasp": primitive_set.GRASP,
            "place_on_top": primitive_set.PLACE_ON_TOP,
            "place_inside": primitive_set.PLACE_INSIDE,
            "open": primitive_set.OPEN,
            "close": primitive_set.CLOSE,
            "release": primitive_set.RELEASE,
            "toggle_on": primitive_set.TOGGLE_ON,
            "toggle_off": primitive_set.TOGGLE_OFF,
        }
        
        if primitive_type not in mapping:
            raise ValueError(
                f"Unknown primitive_type: '{primitive_type}'. "
                f"Available: {list(mapping.keys())}"
            )
        
        return mapping[primitive_type]
    
    # ==================== Task Queue Management ====================
    
    def start_task_sequence(
        self, 
        steps: List[PrimitiveStep],
        on_step_complete: Optional[Callable] = None,
        on_task_complete: Optional[Callable] = None
    ):
        """
        Set and start a task sequence.
        
        This is the public interface for external callers (e.g., PhyPlanPipeline)
        to submit a sequence of primitives.
        
        Args:
            steps: List of PrimitiveStep to execute in order.
            on_step_complete: Callback(step_index, step, success, error) after each step.
            on_task_complete: Callback(success, error) when entire sequence finishes.
            
        Returns:
            bool: True if task sequence started successfully
        """
        if self._task_running:
            rospy.logwarn("Cannot start new task sequence - one is already running")
            return False
        
        if not steps:
            rospy.logwarn("Empty task sequence, nothing to do")
            return False
        
        # Save callbacks
        self._on_step_complete = on_step_complete
        self._on_task_complete = on_task_complete
        
        # Ensure API is initialized
        if self._primitive_api is None:
            if not self._initialize_primitive_api():
                return False
        
        self._task_queue = list(steps)
        self._current_step_index = 0
        self._task_running = True
        
        rospy.loginfo(f"=== Task Sequence Started ({len(steps)} steps) ===")
        for i, step in enumerate(steps):
            rospy.loginfo(f"  [{i+1}] {step.primitive_type}({step.target_name})")
        rospy.loginfo("=" * 50)
        
        # Start the first step
        return self._start_current_step()
    
    def _start_current_step(self) -> bool:
        """
        Start the current step in the task queue by creating its generator.
        
        Returns:
            bool: True if step started successfully
        """
        if self._current_step_index >= len(self._task_queue):
            # All steps completed
            self._finish_task_sequence(success=True)
            return False
        
        step = self._task_queue[self._current_step_index]
        total = len(self._task_queue)
        idx = self._current_step_index + 1
        
        rospy.loginfo(f"--- Step [{idx}/{total}] {step.primitive_type}({step.target_name}) ---")
        
        # Find the target object
        target = self._find_target_object(step.target_name, node_id=step.node_id)
        if target is None:
            rospy.logerr(f"Step [{idx}/{total}] FAILED: target '{step.target_name}' not found")
            self._finish_task_sequence(success=False, error=f"Target '{step.target_name}' not found")
            return False
        
        # Create the generator
        try:
            with og.sim.paused():
                rospy.loginfo(f"Creating generator for {step.primitive_type}({target.name})...")
                if step.primitive_type == "observe":
                    self._primitive_controller = self._create_observe_generator(target)
                else:
                    # Map string type to primitive enum
                    try:
                        primitive_enum = self._get_primitive_enum(step.primitive_type)
                    except ValueError as e:
                        rospy.logerr(f"Step [{idx}/{total}] FAILED: {e}")
                        self._finish_task_sequence(success=False, error=str(e))
                        return False
                    self._primitive_controller = self._primitive_api.controller.apply_ref(
                        primitive_enum,
                        target,
                        attempts=3
                    )
                rospy.loginfo(f"Generator created for step [{idx}/{total}]")
            return True
        except Exception as e:
            rospy.logerr(f"Step [{idx}/{total}] FAILED to create generator: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self._primitive_controller = None
            self._finish_task_sequence(success=False, error=str(e))
            return False
    
    def _advance_to_next_step(self) -> bool:
        """
        Advance to the next step in the task queue.
        Called when the current step's generator raises StopIteration.
        
        Returns:
            bool: True if next step started successfully, False if sequence is done
        """
        idx = self._current_step_index
        total = len(self._task_queue)
        step = self._task_queue[idx]
        
        rospy.loginfo(f"Step [{idx + 1}/{total}] {step.primitive_type}({step.target_name}) COMPLETED")
        
        # Notify step completion (callback may call request_replan())
        if self._on_step_complete:
            try:
                self._on_step_complete(idx, step, True, "")
            except Exception as cb_err:
                rospy.logwarn(f"on_step_complete callback error: {cb_err}")
        
        # Check if callback requested replan (queue replacement)
        if self._replan_steps is not None:
            self._task_queue = self._replan_steps
            self._replan_steps = None
            self._current_step_index = 0
            self._primitive_controller = None
            rospy.loginfo(f"=== Task Queue Replaced: {len(self._task_queue)} new steps ===")
            for i, s in enumerate(self._task_queue):
                rospy.loginfo(f"  [{i+1}] {s.primitive_type}({s.target_name})")
            return self._start_current_step()
        
        self._primitive_controller = None
        self._current_step_index += 1
        
        if self._current_step_index >= len(self._task_queue):
            # All steps done
            self._finish_task_sequence(success=True)
            return False
        
        # Start next step
        return self._start_current_step()
    
    def _finish_task_sequence(self, success: bool, error: str = ""):
        """
        Clean up after task sequence completes or fails.
        
        Args:
            success: Whether the sequence completed successfully
            error: Error message if failed
        """
        total = len(self._task_queue)
        completed = self._current_step_index
        
        self._primitive_controller = None
        self._task_running = False
        
        if success:
            rospy.loginfo(f"=== Task Sequence COMPLETED ({total}/{total} steps) ===")
        else:
            rospy.logerr(f"=== Task Sequence FAILED at step {completed + 1}/{total}: {error} ===")
        
        # Notify task completion
        if self._on_task_complete:
            try:
                self._on_task_complete(success, error)
            except Exception as cb_err:
                rospy.logwarn(f"on_task_complete callback error: {cb_err}")
        
        # 清空队列和回调
        self._task_queue = []
        self._current_step_index = 0
        self._on_step_complete = None
        self._on_task_complete = None
    
    # ==================== Per-Frame Action ====================
    
    def _get_next_action(self):
        """
        Get next action from primitive generator.
        
        Returns:
            action: Robot action array
            
        Raises:
            StopIteration: When generator is exhausted (step completed)
            Exception: On action generation error
        """
        # 直接调用 next()，让 StopIteration 自然传播，不要用 except Exception 捕获它
        return next(self._primitive_controller)
    
    def get_action(self, idle_action):
        """
        Get primitive action for the current frame.
        
        Handles:
        1. G key trigger -> start task sequence
        2. Running task -> get next action from generator
        3. Step complete (StopIteration) -> auto-advance to next step
        4. Step error -> abort task sequence
        5. No task running -> return idle_action
        
        Args:
            idle_action: Fallback action when no primitive is active
            
        Returns:
            action: Robot action array
        """
        # P 键触发：通过 PhyPlanPipeline 交互式规划并执行
        # 规划阶段阻塞（终端交互），执行阶段非阻塞
        if self._p_key_pressed and not self._task_running:
            self._p_key_pressed = False
            if self._pipeline is not None:
                rospy.loginfo("[P key] Starting interactive planning...")
                rospy.loginfo("[P key] (Enter instruction in terminal, "
                              "empty input or 'n' at confirmation to cancel)")
                # instruction=None → run_interactive() 会在终端 prompt 用户输入
                success = self._pipeline.run(instruction=None)
                if not success:
                    rospy.loginfo("[P key] Planning cancelled or failed, staying idle")
                # run() 内部已调用 start_task_sequence()，本帧返回 idle，下帧开始执行
                return idle_action
        
        # G 键触发：始终走硬编码实验序列（用于底层控制器调试，不经过 pipeline）
        if self._g_key_pressed and not self._task_running:
            self._g_key_pressed = False
            rospy.loginfo("G key pressed, starting hardcoded experiment task sequence...")
            self._start_hardcoded_experiment()
            
            if not self._task_running:
                return idle_action
        
        # 如果有 primitive 正在执行，每帧从 generator 取一个 action
        if self._primitive_controller is not None:
            try:
                return self._get_next_action()
            except StopIteration:
                # 当前步骤的 generator 耗尽 -> 步骤完成
                rospy.loginfo("[TaskQueue] Current step generator exhausted")
                try:
                    if self._advance_to_next_step():
                        # 下一步已启动，本帧返回 idle（下一帧开始执行新 generator）
                        return idle_action
                    else:
                        # 任务序列已全部完成或失败
                        return idle_action
                except Exception as e:
                    rospy.logerr(f"[TaskQueue] Error advancing to next step: {e}")
                    import traceback
                    rospy.logerr(traceback.format_exc())
                    self._finish_task_sequence(success=False, error=str(e))
                    return idle_action
            except Exception as e:
                rospy.logwarn(f"Primitive execution error: {e}")
                import traceback
                rospy.logwarn(traceback.format_exc())
                # Notify step failure (callback may call request_replan())
                if self._on_step_complete and self._current_step_index < len(self._task_queue):
                    try:
                        step = self._task_queue[self._current_step_index]
                        self._on_step_complete(self._current_step_index, step, False, str(e))
                    except Exception as cb_err:
                        rospy.logwarn(f"on_step_complete callback error: {cb_err}")
                
                # Check if callback requested replan (failure recovery)
                if self._replan_steps is not None:
                    self._task_queue = self._replan_steps
                    self._replan_steps = None
                    self._current_step_index = 0
                    self._primitive_controller = None
                    rospy.loginfo(f"=== Task Queue Replaced after failure: "
                                  f"{len(self._task_queue)} new steps ===")
                    self._start_current_step()
                    return idle_action
                
                self._finish_task_sequence(
                    success=False, 
                    error=f"Step execution error: {e}"
                )
                return idle_action
        
        return idle_action
