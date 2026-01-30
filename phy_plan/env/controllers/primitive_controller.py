"""
Primitive action controller with progress monitoring.
"""
import rospy
import time
import threading
import psutil
import torch
import gc
import omnigibson as og
from omnigibson.utils.ui_utils import KeyboardEventHandler
import omnigibson.lazy as lazy


class PrimitiveController:
    """Handles semantic action primitives with progress monitoring."""
    
    def __init__(self, env, robot, curobo_batch_size=1):
        """
        Initialize primitive controller.
        
        Args:
            env: OmniGibson environment
            robot: Robot instance
            curobo_batch_size: Batch size for CuRobo (default 1 for 8GB GPU)
        """
        self.env = env
        self.robot = robot
        self.curobo_batch_size = curobo_batch_size
        
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
            
            from phy_plan.executor import BehaviorActionAPI
            
            # 进度提示
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
            
            progress_thread = threading.Thread(target=show_init_progress, daemon=True)
            progress_thread.start()
            
            try:
                # 使用 batch_size=1 来减少 GPU 内存使用（默认是 3，对于 8GB GPU 可能不够）
                rospy.loginfo("Initializing BehaviorActionAPI with curobo_batch_size=1...")
                rospy.loginfo("⚠️  This may take 1-3 minutes. Please wait...")
                
                self._primitive_api = BehaviorActionAPI(
                    self.env, 
                    self.robot,
                    curobo_batch_size=self.curobo_batch_size
                )
                
                # 停止进度提示
                progress_info['active'] = False
                elapsed = time.time() - progress_info['start_time']
                
                rospy.loginfo(f"✅ BehaviorActionAPI initialized successfully in {elapsed:.1f}s (curobo_batch_size=1)")
                if torch.cuda.is_available():
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
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitiveSet,
            )
            
            # 启动 primitive（返回一个 generator）
            rospy.loginfo(f"Starting navigate_to: {target.name}")
            try:
                # 在创建 generator 时暂停仿真（因为可能涉及初始化计算）
                with og.sim.paused():
                    rospy.loginfo("Creating primitive generator (simulation paused)...")
                    self._primitive_controller = self._primitive_api.controller.apply_ref(
                        StarterSemanticActionPrimitiveSet.NAVIGATE_TO,
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
        Get next action from primitive generator with progress monitoring.
        
        Returns:
            action: Robot action array
        """
        # 记录开始时间
        start_time = time.time()
        
        # 创建进度信息字典（内存占用很小，只有几个数值）
        progress_info = {
            'active': True,
            'start_time': start_time,
            'last_gpu_memory': 0.0
        }
        
        # 在后台线程中定期输出详细进度
        def show_progress():
            interval = 2.0  # 每2秒输出一次
            
            while progress_info['active']:
                elapsed = time.time() - progress_info['start_time']
                
                if elapsed > 1.0:  # 超过1秒才开始显示
                    # 获取系统资源使用
                    try:
                        cpu_percent = psutil.cpu_percent(interval=0.1)
                    except:
                        cpu_percent = 0.0
                    
                    # 获取 GPU 使用（如果可用）
                    gpu_info = ""
                    stage_hint = ""
                    if torch.cuda.is_available():
                        try:
                            gpu_memory = torch.cuda.memory_allocated() / 1024**3
                            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                            gpu_info = f"GPU: {gpu_memory:.2f}GB/{gpu_reserved:.2f}GB"
                            
                            # 检测 GPU 内存变化来判断是否在计算
                            if abs(gpu_memory - progress_info['last_gpu_memory']) > 0.01:
                                stage_hint = " (GPU active)"
                                progress_info['last_gpu_memory'] = gpu_memory
                        except:
                            gpu_info = "GPU: N/A"
                    
                    # 估算阶段（基于时间）
                    if elapsed < 5:
                        stage_guess = "Initializing"
                    elif elapsed < 15:
                        stage_guess = "IK solving / Trajectory optimization"
                    elif elapsed < 30:
                        stage_guess = "Collision checking / Refinement"
                    else:
                        stage_guess = "Long computation"
                        stage_hint += " ⚠️"
                    
                    # 组合进度信息
                    progress_msg = (
                        f"[Motion Planning] {stage_guess}{stage_hint} | "
                        f"Time: {elapsed:.1f}s | "
                        f"CPU: {cpu_percent:.1f}%"
                    )
                    
                    if gpu_info:
                        progress_msg += f" | {gpu_info}"
                    
                    rospy.loginfo(progress_msg)
                
                time.sleep(interval)
        
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            # 在获取 action 时暂停仿真（因为可能涉及运动规划计算）
            rospy.loginfo("Getting next action from primitive generator (motion planning may take time, simulation paused)...")
            with og.sim.paused():
                action = next(self._primitive_controller)
            
            # 停止进度提示
            progress_info['active'] = False
            elapsed = time.time() - start_time
            
            # 最终统计
            try:
                final_cpu = psutil.cpu_percent(interval=0.1)
            except:
                final_cpu = 0.0
            
            final_info = f"Completed in {elapsed:.2f}s | CPU: {final_cpu:.1f}%"
            if torch.cuda.is_available():
                try:
                    final_gpu = torch.cuda.memory_allocated() / 1024**3
                    final_info += f" | GPU: {final_gpu:.2f}GB"
                except:
                    pass
            
            rospy.loginfo(f"[Motion Planning] ✅ {final_info} (simulation was paused during computation)")
            
            return action
            
        except Exception as e:
            # 停止进度提示
            progress_info['active'] = False
            elapsed = time.time() - start_time
            rospy.logwarn(f"[Motion Planning] ❌ Failed after {elapsed:.2f}s: {e}")
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
