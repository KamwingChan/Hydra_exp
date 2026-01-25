# Executor Module

This module provides executors for running task plans in BEHAVIOR/Omnigibson simulation.

## Components

### `BehaviorExecutor`
Main executor that takes a `TaskSequence` and executes it in the simulation environment.

**Features:**
- Supports both real BEHAVIOR API and mock execution
- Progress tracking and status updates
- Can load tasks from JSON files
- Detailed logging of execution

**Usage:**

```python
# With real BEHAVIOR API
from phy_plan.executor import BehaviorExecutor
import omnigibson as og

env = og.Environment(configs=config)
executor = BehaviorExecutor(env, use_real_api=True)
executor.execute_task(task_sequence)

# Load from JSON
executor = BehaviorExecutor.from_task_json(env, "output_task_sequence.json")
executor.execute_task(executor.current_task)

# Mock execution (without omnigibson)
executor = BehaviorExecutor(env, use_real_api=False)
```

### `BehaviorActionAPI`
Low-level wrapper around Omnigibson's `StarterSemanticActionPrimitives`.

**Features:**
- Clean error handling (converts exceptions to tuples)
- Structured feedback with metadata
- Convenience methods for common actions

**Available Actions:**
- `grasp(obj)` - Grasp an object
- `place_on_top(surface)` - Place held object on a surface
- `place_inside(container)` - Place held object inside a container
- `navigate_to(obj)` - Navigate to an object
- `open_object(obj)` - Open an object
- `close_object(obj)` - Close an object
- `release()` - Release held object

**Usage:**

```python
from phy_plan.executor import BehaviorActionAPI, StarterSemanticActionPrimitiveSet

api = BehaviorActionAPI(env, robot)

# Using convenience methods
success, message, metadata = api.grasp(target_object)
if not success:
    print(f"Grasp failed: {metadata['reason']}")

# Using primitives directly
success, msg, meta = api.execute_primitive(
    StarterSemanticActionPrimitiveSet.NAVIGATE_TO,
    target_object
)
```

## Action Type Mapping

| LLM Action (phy_plan) | BEHAVIOR Primitive |
|----------------------|-------------------|
| `NAVIGATE` | `NAVIGATE_TO` |
| `PICK` | `GRASP` |
| `PLACE` | `PLACE_ON_TOP` / `PLACE_INSIDE` |
| `OPEN` / `CLOSE` | `OPEN` / `CLOSE` |

## Error Types

When using the BEHAVIOR API, errors are categorized:

- **PRE_CONDITION_ERROR**: Prerequisites not met (e.g., trying to place without holding object)
- **SAMPLING_ERROR**: Failed to find valid configuration (e.g., grasp pose, placement location)
- **PLANNING_ERROR**: Motion planning failed
- **EXECUTION_ERROR**: Error during execution
- **POST_CONDITION_ERROR**: Action completed but postconditions not satisfied

## Dependencies

- **Required**: `phy_plan.core.task`
- **Optional**: `omnigibson` (for real execution; falls back to mock if not available)

## Examples

See:
- `phy_plan/experiments/task_demo.py` - Planning only (generates JSON)
- Future: `env/behavior_execution_demo.py` - Load JSON and execute in simulation
