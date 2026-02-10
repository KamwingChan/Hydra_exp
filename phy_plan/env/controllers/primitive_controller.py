"""
Primitive action controller with progress monitoring.
"""
import rospy
import time
import threading
import torch
import gc
import omnigibson as og
from omnigibson.utils.ui_utils import KeyboardEventHandler
import omnigibson.lazy as lazy


class PrimitiveController:
    """Handles semantic action primitives with progress monitoring."""
    
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
        self._primitive_controller = None  # generator for primitive actions
        self._g_key_pressed = False  # G 键按下标志
        
        # 注册 G 键回调
        try:
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.G,
                self._on_g_key_pressed
            )
            rospy.loginfo("Registered G key callback for primitive control")
        except Exception as e:
            rospy.logwarn(f"Failed to register G key callback: {e}")
    
    def _on_g_key_pressed(self):
        """G 键按下时的回调函数"""
        control_mode = getattr(self, '_current_mode', 'primitive')
        rospy.loginfo(f"G key callback triggered, control_mode={control_mode}, primitive_controller={self._primitive_controller is not None}")
        if control_mode == "primitive" and self._primitive_controller is None:
            self._g_key_pressed = True
            rospy.loginfo("G key pressed - will start primitive in next frame")
        else:
            if control_mode != "primitive":
                rospy.logwarn(f"G key pressed but control_mode is '{control_mode}', not 'primitive'. Switch to primitive mode first!")
            if self._primitive_controller is not None:
                rospy.logwarn("G key pressed but primitive is already running")
    
    def set_mode(self, mode):
        """Set current control mode (for G key callback)."""
        self._current_mode = mode
    
    def _initialize_primitive_api(self):
        """Lazy initialization of BehaviorActionAPI with progress feedback."""
        if self._primitive_api is not None:
            return True
        
        try:
            import time
            import threading
            
            # 在初始化前清理 GPU 缓存和 Python 垃圾，尝试释放一些内存
            gc.collect()  # Python 垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                rospy.loginfo(f"GPU memory before curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                rospy.loginfo(f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
            
            from phy_plan.executor import BehaviorActionAPI, ExecutionMode
            
            # 如果没有指定，默认使用 FULL 模式
            if self.execution_mode is None:
                execution_mode = ExecutionMode.FULL
            else:
                execution_mode = self.execution_mode
            
            # 进度提示（只在 FULL 模式下显示）
            progress_info = {'active': True, 'start_time': time.time()}
            
            def show_init_progress():
                """后台线程显示初始化进度"""
                stages = [
                    (5, "Loading robot configuration..."),
                    (15, "Building collision checker..."),
                    (30, "Initializing motion planners (IK/TrajOpt)..."),
                    (60, "Warming up CUDA kernels (this may take 1-2 minutes)..."),
                ]
                stage_idx = 0
                
                while progress_info['active']:
                    elapsed = time.time() - progress_info['start_time']
                    
                    # 更新阶段提示
                    if stage_idx < len(stages) - 1 and elapsed > stages[stage_idx + 1][0]:
                        stage_idx += 1
                    
                    if elapsed > 3.0:  # 3秒后开始显示
                        stage_msg = stages[min(stage_idx, len(stages) - 1)][1]
                        rospy.loginfo(f"[CuRobo Init] {stage_msg} (elapsed: {elapsed:.1f}s)")
                    
                    time.sleep(5.0)  # 每5秒更新一次
            
            # 只在 FULL 模式下启动进度提示线程
            if execution_mode == ExecutionMode.FULL:
                progress_thread = threading.Thread(target=show_init_progress, daemon=True)
                progress_thread.start()
            
            try:
                # 根据模式显示不同的提示信息
                if execution_mode == ExecutionMode.SYMBOLIC:
                    rospy.loginfo("Initializing BehaviorActionAPI in SYMBOLIC mode (no CuRobo, teleport + physics)...")
                    rospy.loginfo("✅ This will be fast and use minimal GPU memory")
                    motion_cfg_kwargs = None
                else:
                    # 使用 batch_size=1 来减少 GPU 内存使用（默认是 3，对于 8GB GPU 可能不够）
                    rospy.loginfo("Initializing BehaviorActionAPI with curobo_batch_size=1...")
                    rospy.loginfo("⚠️  This may take 1-3 minutes. Please wait...")
                    
                    # 优化的 motion_cfg_kwargs 配置，减少 GPU 内存占用和初始化时间
                    motion_cfg_kwargs = {
                        # 减少 IK seeds（降低内存，稍微降低成功率）
                        'num_ik_seeds': 64,  # 默认 128，减少到 64
                        'num_batch_ik_seeds': 64,  # 默认 128
                        
                        # 减少 trajopt seeds（降低内存）
                        'num_trajopt_seeds': 2,  # 默认 4
                        'num_graph_seeds': 2,  # 默认 4
                        
                        # 减少迭代次数（加快速度，稍微降低质量）
                        'ik_opt_iters': 50,  # 默认 100
                        'finetune_trajopt_iters': 50,  # 默认 100
                        
                        # 减少轨迹优化步数（降低内存）
                        'trajopt_tsteps': 16,  # 默认 32
                    }
                    rospy.loginfo("Using optimized motion_cfg_kwargs to reduce GPU memory usage")
                
                self._primitive_api = BehaviorActionAPI(
                    self.env, 
                    self.robot,
                    mode=execution_mode,  # 传递执行模式
                    curobo_batch_size=self.curobo_batch_size,
                    motion_cfg_kwargs=motion_cfg_kwargs
                )
                
                # 停止进度提示
                if execution_mode == ExecutionMode.FULL:
                    progress_info['active'] = False
                elapsed = time.time() - progress_info['start_time']
                
                mode_str = "SYMBOLIC" if execution_mode == ExecutionMode.SYMBOLIC else "FULL"
                rospy.loginfo(f"✅ BehaviorActionAPI initialized successfully in {elapsed:.1f}s (mode: {mode_str})")
                if torch.cuda.is_available() and execution_mode == ExecutionMode.FULL:
                    rospy.loginfo(f"GPU memory after curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                return True
            except Exception as e:
                progress_info['active'] = False
                raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                rospy.logerr(f"GPU out of memory when initializing curobo. Current GPU usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                rospy.logerr("Suggestions:")
                rospy.logerr("  1. Close other GPU-intensive applications")
                rospy.logerr("  2. Reduce scene complexity")
                rospy.logerr("  3. Restart the simulation to free GPU memory")
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
    
    def _find_target_object(self, target_name="coffee"):
        """
        Find target object in scene.
        
        Args:
            target_name: Name or category keyword to search for
            
        Returns:
            target: Object instance or None
        """
        # 方法1: 通过名字查找
        target = None
        for obj in self.env.scene.objects:
            if hasattr(obj, 'name') and target_name.lower() in obj.name.lower():
                target = obj
                rospy.loginfo(f"Found target by name: {obj.name}")
                return target
        
        # 方法2: 如果没找到，通过 category 查找
        for obj in self.env.scene.objects:
            if hasattr(obj, 'category') and target_name.lower() in str(obj.category).lower():
                target = obj
                rospy.loginfo(f"Found target by category: {obj.name} (category: {obj.category})")
                return target
        
        # 方法3: 如果还是没找到，列出所有对象
        rospy.logwarn(f"Target '{target_name}' not found. Available objects:")
        for obj in self.env.scene.objects:
            if hasattr(obj, 'name'):
                rospy.logwarn(f"  - {obj.name} (category: {getattr(obj, 'category', 'N/A')})")
        
        return None
    
    def _start_primitive(self, target):
        """Start a primitive action on target object."""
        if self._primitive_api is None:
            if not self._initialize_primitive_api():
                return False
        
        try:
            # 使用 BehaviorActionAPI 中保存的正确的 PrimitiveSet
            primitive_set = self._primitive_api._primitive_set
            
            # 启动 primitive（返回一个 generator）
            rospy.loginfo(f"Starting navigate_to: {target.name}")
            try:
                # 在创建 generator 时暂停仿真（因为可能涉及初始化计算）
                with og.sim.paused():
                    rospy.loginfo("Creating primitive generator (simulation paused)...")
                    self._primitive_controller = self._primitive_api.controller.apply_ref(
                        primitive_set.NAVIGATE_TO,
                        target,
                        attempts=3
                    )
                    rospy.loginfo("Primitive generator created successfully")
                return True
            except Exception as e:
                rospy.logerr(f"Failed to create primitive generator: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                self._primitive_controller = None
                return False
        except Exception as e:
            rospy.logwarn(f"Failed to start navigate_to: {e}")
            import traceback
            rospy.logwarn(traceback.format_exc())
            self._primitive_controller = None
            return False
    
    def _get_action_with_progress(self):
        """
        Get next action from primitive generator.
        
        NOTE: 不使用 og.sim.paused()，因为每帧的 pause→play 循环会触发
        play() → render() → update_handles() → _non_physics_step()，
        造成不必要的开销和潜在的 FlatCache 数据不一致。
        generator 只是读取当前位姿并计算速度命令，不需要暂停仿真。
        
        Returns:
            action: Robot action array
        """
        try:
            # 直接获取下一帧的 action（无需暂停仿真）
            action = next(self._primitive_controller)
            return action
        except Exception as e:
            rospy.logwarn(f"[Primitive] Action generation failed: {e}")
            raise
    
    def get_action(self, idle_action):
        """
        Get primitive action.
        
        Args:
            idle_action: Fallback action if primitive fails
            
        Returns:
            action: Robot action array
        """
        # 如果按了 G 键，开始执行 navigate_to coffee cup
        if self._g_key_pressed and self._primitive_controller is None:
            rospy.loginfo("G key pressed, starting navigate_to primitive...")
            self._g_key_pressed = False  # 重置标志
            
            target = self._find_target_object("coffee")
            if target is None:
                return idle_action
            
            if not self._start_primitive(target):
                return idle_action
        
        # 如果 primitive 正在执行，每帧从 generator 取一个 action
        if self._primitive_controller is not None:
            try:
                return self._get_action_with_progress()
            except StopIteration:
                # primitive 执行完成
                rospy.loginfo("Primitive execution completed")
                self._primitive_controller = None
                return idle_action
            except Exception as e:
                rospy.logwarn(f"Primitive execution error: {e}")
                import traceback
                rospy.logwarn(traceback.format_exc())
                self._primitive_controller = None
                return idle_action
        
        return idle_action
