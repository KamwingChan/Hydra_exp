# phy_plan 高层规划框架

## 项目核心 Contribution

**Dynamic Scene Graph + Physics-Aware Planning**

- 动态场景图实时更新（5s周期）
- 物理属性感知的任务规划（力气 vs weight_level）
- 多轮对话支持动态重规划

## 架构概览

```mermaid
flowchart LR
    subgraph Input [输入层]
        A[语言指令]
        B[phy_graph JSON<br/>5s实时更新]
        C[Hydra DSG]
    end
    
    subgraph Core [核心处理]
        D[统一 SceneGraph 接口<br/>compact + 详细版本]
        E[物理感知 Agent<br/>力气评估]
        F[LLM Planner<br/>多轮对话]
    end
    
    subgraph Output [输出层]
        G[TaskSequence]
        H[可视化]
    end
    
    A --> F
    B -->|phy_graph_io| D
    C -->|hydra_io| D
    D -->|compact JSON| F
    D -->|详细信息检索| E
    E -->|物理约束| F
    F --> G
    G --> H
```

## 目录结构

```
phy_plan/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── agent.py              # ✅ LLM Agent 封装（单轮/多轮对话）
│   ├── scene_graph.py        # ✅ 统一接口 + to_compact_json() + to_verbose_description()
│   └── task.py               # ✅ 领域语言（Action, TaskSequence）
├── input/
│   ├── __init__.py
│   ├── phy_graph_io.py       # ✅ 加载 phy_graph JSON
│   └── hydra_io.py           # ✅ 加载 Hydra DSG（含 GVD places）
├── planner/
│   ├── __init__.py
│   ├── llm_planner.py        # ✅ LLM 规划器
│   └── llm_planner_pipeline.py # ✅ 完整 Pipeline
├── prompts/
│   ├── __init__.py
│   └── task_planning_prompt.py # ✅ Prompt 模板（few-shot + 结构化输出）
├── output/
│   ├── __init__.py
│   └── task_sequence.py
├── visualization/
│   ├── __init__.py
│   └── arrangement_viz.py    # ✅ 2D 可视化 + 动画
├── utils/
│   ├── assignment.py         # ✅ 匈牙利算法 + TSP
│   └── arrangement.py        # ✅ 位置生成
├── experiments/
│   ├── chair_arrangement.py  # ✅ 椅子摆放实验
│   ├── task_demo.py          # ✅ LLM 规划 Demo
│   └── data/
│       └── test_scene_coffee.json # ✅ 测试场景
└── test/                     # 保留
```

## Compact vs 详细版本设计

**设计目的**：节省 LLM token，同时支持详细信息检索

```
Compact 版本（用于 LLM prompt）     详细版本（用于检索）
─────────────────────────────────────────────────────────
{                                   {
  "node_id": "O(13)",                "node_id": "O(13)",
  "category": "chair",               "category": "chair",
  "room_id": "R(2)"                  "position": {"x": 1.5, "y": 2.0, "z": 0.5},
}                                    "bounding_box": {...},
                                     "physical_properties": {
                                       "weight_level": 1,
                                       "pushable": true,
                                       "friction_level": 1
                                     }
                                   }
```

**流程**：
1. `to_compact_json()` 生成 prompt → LLM 输出引用 `node_id`
2. 解析后从完整 SceneGraph 检索 `position`, `bounding_box`, `physical_properties`

## 快速开始

```bash
# 离线测试（不调用 LLM）
cd /home/kamwing/catkin_ws/src/phy_plan
python phy_plan/experiments/task_demo.py --offline

# 完整测试
export OPENAI_API_KEY='your-key'
python phy_plan/experiments/task_demo.py \
    --scene phy_plan/experiments/data/test_scene_coffee.json \
    --instruction "去小房间拿咖啡杯到大会议室，然后整理椅子"
```

---

## 实现路线图

### ✅ Phase 0: 基础框架（已完成）

- [x] 单轮 LLM 调用 (`core/agent.py`)
- [x] Compact JSON 生成 (`scene_graph.to_compact_json()`)
- [x] Prompt 模板 + few-shot (`prompts/task_planning_prompt.py`)
- [x] LLM Planner + Pipeline (`planner/`)
- [x] 离线测试 Demo (`experiments/task_demo.py`)
- [x] 椅子摆放实验 (`experiments/chair_arrangement.py`)

### 🔄 Phase 1: 多轮对话支持（近期）

**目标**：支持任务执行中的动态重规划

```python
# core/agent.py 扩展
class LLMAgent:
    def __init__(self):
        self.prompt_chain = []  # 对话历史
        
    def init_conversation(self, system, user):
        """初始化对话"""
        
    def continue_conversation(self, user_msg):
        """继续对话（带上下文）"""
        
    def reset(self):
        """重置对话"""
```

### 🔄 Phase 2: 实时 phy_graph 对接（近期）

**目标**：5秒周期订阅 phy_graph 更新

```python
# input/phy_graph_realtime.py
class PhyGraphSubscriber:
    def __init__(self, update_interval=5.0):
        self.scene_graph = None
        self.last_update = None
        
    def subscribe(self, callback):
        """订阅场景图更新"""
        # 选项 A: 监听文件变化
        # 选项 B: ROS topic（推荐）
        
    def get_latest(self) -> SceneGraph:
        """获取最新场景图"""
```

### 🔄 Phase 3: 物理感知 Agent（核心创新）

**目标**：机器人力气评估 + 物理约束检查

```python
# core/physics_agent.py
@dataclass
class RobotCapability:
    max_force: int = 2          # 最大力气等级 0-2
    gripper_width: float = 0.1  # 夹持宽度
    reach_distance: float = 1.0 # 可达距离

class PhysicsAwareAgent:
    def __init__(self, capability: RobotCapability):
        self.capability = capability
        
    def can_manipulate(self, obj: ObjectNode) -> bool:
        """判断能否操作物体"""
        if obj.physical_properties:
            return obj.physical_properties.weight_level <= self.capability.max_force
        return True
        
    def filter_feasible_actions(self, actions, sg) -> List[Action]:
        """过滤物理上可行的动作"""
```

**物理约束**：
- `weight_level > robot.max_force` → 无法搬运
- `pushable == False` → 无法推动
- `friction_level` → 影响推动策略

### 🔄 Phase 4: 动态重规划 Pipeline（核心创新）

**目标**：场景变化时自动重规划

```python
# planner/dynamic_planner.py
class DynamicPlannerPipeline:
    def __init__(self, agent, phy_graph_sub, robot_capability):
        self.agent = agent
        self.phy_graph_sub = phy_graph_sub
        self.physics_agent = PhysicsAwareAgent(robot_capability)
        
    def plan_and_monitor(self, instruction):
        """规划并监控执行"""
        while not self.is_complete():
            # 1. 检查场景图是否有重大变化
            if self.scene_changed_significantly():
                # 2. 多轮对话重规划
                self.replan_with_context()
            # 3. 物理约束检查
            self.physics_agent.filter_feasible_actions(...)
            # 4. 执行下一个动作
            self.execute_next_action()
```

### 🔮 Future: GVD 导航路径规划

利用 Hydra DSG 的 Places 层进行导航路径规划（已有 `hydra_io.py` 支持）

---

## 依赖关系

| 阶段 | 依赖 | 状态 |
|------|------|------|
| Phase 0 | - | ✅ 完成 |
| Phase 1 | Phase 0 | 🔄 待实现 |
| Phase 2 | Phase 1 | 🔄 待实现 |
| Phase 3 | Phase 1 | 🔄 待实现 |
| Phase 4 | Phase 1-3 | 🔄 待实现 |

---

## 参考项目

- **context-matters**: Pipeline 架构、GPTAgent prompt_chain 设计
- **phy_graph**: 物理属性推断、compact/detailed JSON 格式
- **Hydra**: DSG 场景图、GVD places 图
