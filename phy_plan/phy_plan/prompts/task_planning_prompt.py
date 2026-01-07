"""
task_planning_prompt.py: 任务规划 Prompt 模板

定义 LLM 任务规划的 system content 和 user prompt 模板。
"""

from typing import Optional, Tuple


# System Content: 定义 LLM 角色和能力
SYSTEM_CONTENT = """You are an expert robot task planner. Given a 3D scene graph and a natural language instruction, you generate a step-by-step task plan.

## Available Actions
The robot can perform these actions:
1. navigate(room_id): Move to a specific room
2. pick(object_id): Pick up an object (robot must be in the same room)
3. place(object_id, surface_id): Place the held object ON a surface object (e.g., table, desk, shelf)
   - surface_id: The node_id of the surface object where you want to place the item
   - Always prefer placing on a surface rather than just in a room
4. arrange(object_category, room_id): Arrange objects of a category in a room (e.g., align chairs around tables)

## Output Format
You MUST respond with a valid JSON object containing:
{
    "chain_of_thought": "Your step-by-step reasoning",
    "plan": [
        {"action": "navigate", "params": {"room_id": "R(x)"}},
        {"action": "pick", "params": {"object_id": "O(x)"}},
        {"action": "place", "params": {"object_id": "O(x)", "surface_id": "O(table_id)"}},
        {"action": "arrange", "params": {"object_category": "category_name", "room_id": "R(x)"}}
    ]
}

## Rules
1. Always navigate to the room containing the target object before picking it up
2. Use exact node_id (e.g., "O(13)") and room_id (e.g., "R(2)") from the scene graph
3. The plan must be finite and executable
4. For arrange actions, specify the object category (e.g., "chair", "swivel_chair")
"""


# Few-shot Example
FEW_SHOT_EXAMPLE = """
## Example

**Scene Graph (compact):**
{
  "rooms": [
    {"room_id": "R(0)", "category": "SmallRoom", "object_ids": ["O(5)"]},
    {"room_id": "R(1)", "category": "ConferenceRoom", "object_ids": ["O(10)", "O(11)", "O(12)"]}
  ],
  "objects": [
    {"node_id": "O(5)", "category": "coffee_cup", "room_id": "R(0)"},
    {"node_id": "O(10)", "category": "swivel_chair", "room_id": "R(1)"},
    {"node_id": "O(11)", "category": "swivel_chair", "room_id": "R(1)"},
    {"node_id": "O(12)", "category": "conference_table", "room_id": "R(1)"}
  ]
}

**Instruction:** Go to the small room, pick up the coffee cup, bring it to the conference room, and arrange the chairs.

**Output:**
{
    "chain_of_thought": "1. The coffee_cup O(5) is in SmallRoom R(0). 2. First navigate to R(0). 3. Pick up the coffee_cup O(5). 4. Navigate to ConferenceRoom R(1) where the conference_table O(12) is. 5. Place the coffee_cup ON the conference_table O(12). 6. Arrange swivel_chairs in R(1) around the table.",
    "plan": [
        {"action": "navigate", "params": {"room_id": "R(0)"}},
        {"action": "pick", "params": {"object_id": "O(5)"}},
        {"action": "navigate", "params": {"room_id": "R(1)"}},
        {"action": "place", "params": {"object_id": "O(5)", "surface_id": "O(12)"}},
        {"action": "arrange", "params": {"object_category": "swivel_chair", "room_id": "R(1)"}}
    ]
}
"""


def generate_task_planning_prompt(
    scene_graph_compact: str,
    instruction: str,
    include_example: bool = True
) -> Tuple[str, str]:
    """
    生成任务规划的 prompt
    
    Args:
        scene_graph_compact: Compact 格式的场景图 JSON
        instruction: 用户指令
        include_example: 是否包含 few-shot 示例
        
    Returns:
        (system_content, user_prompt) 元组
    """
    system = SYSTEM_CONTENT
    if include_example:
        system += FEW_SHOT_EXAMPLE
    
    user_prompt = f"""## Current Task

**Scene Graph (compact):**
{scene_graph_compact}

**Instruction:** {instruction}

Please generate the task plan as a JSON object. Use ONLY the node_ids and room_ids from the scene graph above."""
    
    return system, user_prompt


# Output Format Definition (for parsing)
OUTPUT_FORMAT = {
    "chain_of_thought": "string - reasoning steps",
    "plan": [
        {
            "action": "navigate | pick | place | arrange",
            "params": {
                "room_id": "optional - target room",
                "object_id": "optional - target object",
                "object_category": "optional - for arrange action"
            }
        }
    ]
}
