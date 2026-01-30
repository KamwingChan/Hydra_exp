import omnigibson as og
from omnigibson.robots import Stretch


def main():
    # 1. 建一个最简单的空场景
    cfg = {
        "scene": {
            "type": "Scene",
        }
    }
    env = og.Environment(configs=cfg)

    # 2. 先停一下 sim，按官方示例的用法
    og.sim.stop()

    # 3. 创建一个 Fetch，并打开你关心的模态，同时指定一个不太遮挡视野的默认手臂姿态
    robot = Stretch(
        name="stretch0",
        obs_modalities=("rgb", "depth", "seg_semantic", ),
        default_reset_mode="untuck",
        default_arm_pose="horizontal",  # 可以试试 "diagonal30" / "diagonal45" 看遮挡情况
    )
    env.scene.add_object(robot)

    # 4. 让 simulator 跑一小步，让机器人真正初始化完成
    og.sim.play()
    og.sim.step()

    # 5. 重置机器人并静止
    robot.reset()
    robot.keep_still()

    # 6. 打印传感器名字（关键：用来看相机都叫什么）
    print("=== Fetch sensors ===")
    try:
        for name in robot._sensors.keys():
            print("  ", name)
    except Exception as e:
        print("List sensors failed:", e)

    # 7. 再打印每个传感器有哪些 modality（rgb / depth / seg_semantic）
    try:
        obs_dict, info_dict = robot.get_obs()
        print("=== Fetch sensor obs modalities ===")
        for sensor_name, d in obs_dict.items():
            if isinstance(d, dict):
                print("  ", sensor_name, "->", list(d.keys()))
            else:
                print("  ", sensor_name, "->", type(d))
    except Exception as e:
        print("Get obs failed:", e)

    input("Press Enter to quit...")
    og.shutdown()


if __name__ == "__main__":
    main()