#!/usr/bin/env python3
"""
精确地图生成工具 - 使用 OmniGibson AABB 获取真实物体尺寸

从场景 JSON 文件生成可通行地图（Traversability Map），输出格式与 OmniGibson
TraversableMap 一致：正方形、地图中心=世界(0,0)、0.01 m/px、255=可通行/0=障碍、
地图外（场景 bbox 外）为黑色。可选 --use-mesh-projection：用 mesh 俯视投影（保留门洞），
仅投影 1.5m 以下且排除 floor 的几何。

用法:
    python generate_trav_map_accurate.py \\
        --scene /path/to/scene.json \\
        --output floor_trav_0.png \\
        --resolution 0.01
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# 添加 OmniGibson 到路径
try:
    import torch as th
    import omnigibson as og
    from omnigibson.utils.ui_utils import choose_from_options
    from omnigibson.macros import gm
except ImportError:
    print("错误: 无法导入 OmniGibson. 请确保已安装 OmniGibson")
    sys.exit(1)


class AccurateTravMapGenerator:
    """
    精确可通行地图生成器。
    障碍物：只画地面上的物体（AABB 底面 z 低于 floor_height_threshold 的物体画其俯视投影），
    但排除“地板/天花板”等可通行表面，否则整张图会几乎全黑。
    输出与 OmniGibson TraversableMap 兼容：正方形、中心=世界(0,0)、外边黑色。
    """

    # 不画为障碍的类别：地板、天花板等表示可通行表面的物体，画进去会导致地图几乎全黑
    SKIP_OBSTACLE_CATEGORIES = {"floors", "ceilings", "carpet"}

    def __init__(
        self,
        scene_json_path,
        resolution=0.01,
        margin=1.0,
        safety_padding=0.0,
        floor_height_threshold=0.3,
        scene_model=None,
        use_mesh_projection=False,
        mesh_height_max=1.5,
    ):
        """
        Args:
            scene_json_path: 场景 JSON 文件路径
            resolution: 地图分辨率（米/像素），默认 0.01
            margin: 场景边界额外边距（米），默认 1.0
            safety_padding: 障碍物安全边距（米），默认 0.0（精确AABB）
            floor_height_threshold: AABB 底面 z 低于此值才视为“地面上”并画为障碍（米），默认 0.3
            scene_model: OmniGibson 场景名（如 Rs_int）；若不传则从 --scene 路径中的 .../scenes/<name>/... 推断
            use_mesh_projection: 若 True，用 mesh 俯视投影（保留门洞等），仅投影 z in [0, mesh_height_max] 的几何，排除 floor
            mesh_height_max: mesh 投影时只考虑此高度以下的三角形（米），默认 1.5
        """
        self.scene_json_path = os.path.abspath(scene_json_path)
        self.resolution = resolution
        self.margin = margin
        self.safety_padding = safety_padding
        self.floor_height_threshold = floor_height_threshold
        self.scene_model = scene_model
        self.use_mesh_projection = use_mesh_projection
        self.mesh_height_max = mesh_height_max

        self.env = None
        self.trav_map = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.map_size = None
        self._half_side_m = None
        
    def _resolve_scene_model(self):
        """解析 scene_model：若未传入则从路径 .../scenes/<name>/... 推断"""
        if self.scene_model:
            return self.scene_model
        path = self.scene_json_path.replace("\\", "/")
        if "/scenes/" in path:
            after = path.split("/scenes/")[-1]
            name = after.split("/")[0]
            if name and name != "json":
                return name
        raise ValueError(
            "无法从路径推断 OmniGibson 场景名。请用 --scene-model 指定，例如: "
            "--scene-model Rs_int（路径通常形如 .../scenes/Rs_int/json/xxx.json）"
        )

    def _get_mesh_triangles_in_band(self, obj):
        """
        收集物体在 z in [0, mesh_height_max] 内的 mesh 三角形（世界坐标），用于俯视投影。
        返回 list of (3, 3) numpy array，每个为三角形三个顶点的 xyz。
        """
        triangles = []
        links = getattr(obj, "_links", None)
        if links is None:
            links = [obj]
        else:
            links = list(links.values())
        for link in links:
            visual = getattr(link, "visual_meshes", None) or getattr(link, "_visual_meshes", {})
            collision = getattr(link, "collision_meshes", None) or getattr(link, "_collision_meshes", {})
            for mesh in list(visual.values()) + list(collision.values()):
                try:
                    pts = mesh.points
                    if pts is None:
                        continue
                    faces = mesh.faces
                    if faces is None:
                        continue
                    world_pose = mesh.scaled_transform
                    if world_pose is None:
                        continue
                    pts = th.as_tensor(pts, device=world_pose.device, dtype=world_pose.dtype)
                    pts_h = th.cat([pts, th.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)], dim=1)
                    world_pts = (pts_h @ world_pose.T)[:, :3]
                    world_pts = world_pts.cpu().numpy()
                    face_idx = faces.cpu().numpy()
                    for i in range(len(face_idx)):
                        tri = world_pts[face_idx[i]]
                        z_min, z_max = float(tri[:, 2].min()), float(tri[:, 2].max())
                        if z_max < 0 or z_min > self.mesh_height_max:
                            continue
                        triangles.append(tri)
                except Exception:
                    continue
        return triangles

    def load_scene(self):
        """加载 OmniGibson 场景"""
        print(f"正在加载场景: {self.scene_json_path}")
        scene_model = self._resolve_scene_model()
        print(f"场景名 (scene_model): {scene_model}")

        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_OBJECT_STATES = False
        gm.ENABLE_TRANSITION_RULES = False

        config = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
                "scene_file": self.scene_json_path,
            },
            "robots": [],
        }

        print("初始化 OmniGibson 环境（可能需要 10-30 秒）...")
        self.env = og.Environment(configs=config)
        # 传入 scene_file 时，场景已在 Environment 初始化时从该 JSON 加载，无需再 load(scene_data)

        print(f"✅ 场景加载完成，共 {len(self.env.scene.objects)} 个物体")
        
    def calculate_bounds(self):
        """计算场景边界"""
        print("计算场景边界...")
        
        positions = []
        for obj in self.env.scene.objects:
            try:
                pos = obj.get_position_orientation()[0]
                positions.append([pos[0], pos[1]])
            except:
                continue
        
        if not positions:
            raise ValueError("场景中没有找到有效的物体位置")
        
        positions = np.array(positions)
        self.min_x = float(positions[:, 0].min() - self.margin)
        self.min_y = float(positions[:, 1].min() - self.margin)
        self.max_x = float(positions[:, 0].max() + self.margin)
        self.max_y = float(positions[:, 1].max() + self.margin)

        # OmniGibson 约定：正方形、地图中心 = 世界 (0,0)
        half_side_m = max(
            abs(self.min_x), abs(self.max_x), abs(self.min_y), abs(self.max_y)
        ) + self.margin
        self._half_side_m = half_side_m
        self.map_size = max(1, int(2.0 * half_side_m / self.resolution))

        print(f"场景范围: X[{self.min_x:.2f}, {self.max_x:.2f}], Y[{self.min_y:.2f}, {self.max_y:.2f}]")
        print(f"地图: 正方形 {self.map_size}×{self.map_size} px, 中心=世界(0,0), 边长 {self.map_size * self.resolution:.2f} m")
        
    def generate_map(self):
        """生成可通行地图（正方形、中心=世界0,0；只画地面上的物体；外边黑色）"""
        c = self.map_size / 2.0

        # 创建全白正方形地图（可通行）
        self.trav_map = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 255

        if self.use_mesh_projection:
            print("处理物体（mesh 投影，仅 z in [0, {:.1f}m] 且排除 floor/ceiling）...".format(self.mesh_height_max))
        else:
            print("处理物体（仅地面上的物体作为障碍，AABB）...")
        obstacle_count = 0
        skipped_count = 0

        for obj in self.env.scene.objects:
            try:
                category = getattr(obj, "category", "") or ""
                if any(skip in category for skip in self.SKIP_OBSTACLE_CATEGORIES):
                    skipped_count += 1
                    continue

                if self.use_mesh_projection:
                    # Mesh 投影：只投影 1.5m 以下的三角形，门洞等会保留
                    tri_list = self._get_mesh_triangles_in_band(obj)
                    if not tri_list:
                        skipped_count += 1
                        continue
                    for tri in tri_list:
                        xy = tri[:, :2]
                        col_row = np.zeros((3, 2), dtype=np.int32)
                        col_row[:, 0] = (xy[:, 0] / self.resolution + c).astype(np.int32)
                        col_row[:, 1] = (xy[:, 1] / self.resolution + c).astype(np.int32)
                        col_row[:, 0] = np.clip(col_row[:, 0], 0, self.map_size - 1)
                        col_row[:, 1] = np.clip(col_row[:, 1], 0, self.map_size - 1)
                        cv2.fillPoly(self.trav_map, [col_row], 0)
                    obstacle_count += 1
                else:
                    # AABB 模式
                    center = np.asarray(obj.aabb_center).flatten()
                    extent = np.asarray(obj.aabb_extent).flatten()
                    bottom_z = float(center[2] - extent[2])
                    if bottom_z > self.floor_height_threshold:
                        skipped_count += 1
                        continue
                    center_xy = center[:2]
                    extent_xy = extent[:2]
                    extent_with_padding = extent_xy + 2 * self.safety_padding
                    row_center = center_xy[1] / self.resolution + c
                    col_center = center_xy[0] / self.resolution + c
                    map_w = extent_with_padding[0] / self.resolution
                    map_h = extent_with_padding[1] / self.resolution
                    col1 = max(0, int(col_center - map_w / 2))
                    row1 = max(0, int(row_center - map_h / 2))
                    col2 = min(self.map_size, int(col_center + map_w / 2))
                    row2 = min(self.map_size, int(row_center + map_h / 2))
                    cv2.rectangle(self.trav_map, (col1, row1), (col2, row2), 0, -1)
                    obstacle_count += 1

                if obstacle_count % 10 == 0:
                    print(f"  已处理 {obstacle_count} 个障碍物...")
            except Exception as e:
                print(f"  警告: 处理物体 {getattr(obj, 'name', '?')} 时出错: {e}")
                skipped_count += 1
                continue

        # 外边用黑色填充：场景 bbox 外的像素设为不可通行
        cols = np.arange(self.map_size, dtype=np.float64)
        rows = np.arange(self.map_size, dtype=np.float64)
        wx_grid = (cols - c) * self.resolution
        wy_grid = (rows - c) * self.resolution
        wx_grid, wy_grid = np.meshgrid(wx_grid, wy_grid)
        outside = (
            (wx_grid < self.min_x) | (wx_grid > self.max_x)
            | (wy_grid < self.min_y) | (wy_grid > self.max_y)
        )
        self.trav_map[outside] = 0
        mode = "mesh 投影" if self.use_mesh_projection else "AABB"
        print(f"✅ 地图生成完成 ({mode}): {obstacle_count} 个障碍物, {skipped_count} 个跳过, 外边已填黑")
    
    def save_map(self, output_path):
        """保存地图到文件"""
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存地图
        cv2.imwrite(output_path, self.trav_map)
        print(f"✅ 地图已保存到: {output_path}")
        
        # 生成预览图（带网格和统计信息）
        self._save_preview(output_path)
        
    def _save_preview(self, original_path):
        """生成带网格和统计信息的预览图"""
        preview_path = original_path.replace('.png', '_preview.png')
        
        # 创建彩色预览图
        preview = cv2.cvtColor(self.trav_map, cv2.COLOR_GRAY2BGR)
        
        # 计算统计信息
        total_pixels = self.trav_map.size
        traversable_pixels = np.sum(self.trav_map == 255)
        obstacle_pixels = np.sum(self.trav_map == 0)
        traversable_percent = (traversable_pixels / total_pixels) * 100
        
        # 添加网格（每米一条线）
        grid_interval = max(1, int(1.0 / self.resolution))
        height, width = self.trav_map.shape
        for i in range(0, width, grid_interval):
            cv2.line(preview, (i, 0), (i, height), (200, 200, 200), 1)
        for i in range(0, height, grid_interval):
            cv2.line(preview, (0, i), (width, i), (200, 200, 200), 1)

        side_m = self.map_size * self.resolution
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = [
            f"Size: {width}x{height} px (square)",
            f"Area: {side_m:.1f}x{side_m:.1f} m, center=world(0,0)",
            f"Resolution: {self.resolution} m/px",
            f"Traversable: {traversable_percent:.1f}%",
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(preview, text, (10, y_offset), font, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.imwrite(preview_path, preview)
        print(f"📊 预览图已保存到: {preview_path}")
        print(f"   可通行区域: {traversable_percent:.1f}%")
        
    def cleanup(self):
        """清理资源"""
        if self.env is not None:
            print("关闭 OmniGibson 环境...")
            og.shutdown()
            
    def generate(self, output_path):
        """完整的地图生成流程"""
        try:
            self.load_scene()
            self.calculate_bounds()
            self.generate_map()
            self.save_map(output_path)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            raise
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="精确可通行地图生成工具（基于 OmniGibson AABB）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python generate_trav_map_accurate.py \\
      --scene ../config/scene_configs/office_vendor_machine_0.json \\
      --output floor_trav_0.png

  # 自定义分辨率和边距
  python generate_trav_map_accurate.py \\
      --scene scene.json \\
      --output map.png \\
      --resolution 0.02 \\
      --margin 2.0 \\
      --safety-padding 0.1

在导航时使用生成的地图:
  # 方式一：启动仿真时指定 --trav_map（推荐）
  python behavior_ros_robot.py --scene office_vendor_machine --scene_file ... --trav_map /path/to/floor_trav_0.png

  # 方式二：将 floor_trav_0.png 放到场景 layout 目录，OmniGibson 会自动加载
  例如: <dataset>/scenes/<scene_name>/layout/floor_trav_0.png
        """
    )
    
    parser.add_argument(
        '--scene',
        type=str,
        default='/home/kamwing/catkin_ws/src/phy_plan/env/config/scene_configs/office_vendor_machine_0.json',
        help='场景 JSON 文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='floor_trav_0.png',
        help='输出地图文件路径（默认: floor_trav_0.png）'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.01,
        help='地图分辨率（米/像素），默认: 0.01 (与官方一致)'
    )
    
    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='场景边界额外边距（米），默认: 1.0'
    )
    
    parser.add_argument(
        '--safety-padding',
        type=float,
        default=0.0,
        help='障碍物安全边距（米），默认: 0.0（精确AABB）'
    )
    parser.add_argument(
        '--floor-height-threshold',
        type=float,
        default=0.3,
        help='AABB 底面 z 低于此值才视为地面上的物体并画为障碍（米），默认: 0.3'
    )
    parser.add_argument(
        '--scene-model',
        type=str,
        default="office_vendor_machine",
        help='OmniGibson 场景名（如 Rs_int）。若不传则从 --scene 路径中的 .../scenes/<name>/... 自动推断'
    )
    parser.add_argument(
        '--use-mesh-projection',
        action='store_true',
        help='用 mesh 俯视投影代替 AABB，保留门洞等；仅投影 z in [0, mesh-height-max] 的几何，排除 floor'
    )
    parser.add_argument(
        '--mesh-height-max',
        type=float,
        default=1.5,
        help='mesh 投影时只考虑此高度以下的三角形（米），默认 1.5'
    )

    args = parser.parse_args()
    
    # 检查场景文件是否存在
    if not os.path.exists(args.scene):
        print(f"错误: 场景文件不存在: {args.scene}")
        sys.exit(1)
    
    print("=" * 60)
    print("精确可通行地图生成工具")
    print("=" * 60)
    print(f"场景文件: {args.scene}")
    print(f"输出文件: {args.output}")
    print(f"分辨率: {args.resolution} m/px")
    print(f"边距: {args.margin} m")
    print(f"安全边距: {args.safety_padding} m")
    print(f"地面高度阈值: {args.floor_height_threshold} m")
    if args.scene_model:
        print(f"场景名 (--scene-model): {args.scene_model}")
    if args.use_mesh_projection:
        print(f"Mesh 投影: 开, 高度上限: {args.mesh_height_max} m")
    print("=" * 60)

    generator = AccurateTravMapGenerator(
        scene_json_path=args.scene,
        resolution=args.resolution,
        margin=args.margin,
        safety_padding=args.safety_padding,
        floor_height_threshold=args.floor_height_threshold,
        scene_model=args.scene_model,
        use_mesh_projection=args.use_mesh_projection,
        mesh_height_max=args.mesh_height_max,
    )
    
    generator.generate(args.output)
    
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
