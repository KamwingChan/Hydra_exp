from graphReader import DsgVisualizer, DsgStatistics, load_dsg, DsgQuery, NodeInfo, visualize_gvd_graph
from spark_dsg import NodeSymbol
import spark_dsg as dsg
dsg_file = "backend/dsg.json"
mesh_file = "backend/mesh.ply"

graph = load_dsg(dsg_file, mesh_file)
# print("场景图加载成功！\n")

# # 显示统计信息
# print("场景图统计信息:")
# print("-" * 60)
# statistics = DsgStatistics(graph)
# statistics.print_summary()

# 创建查询对象
statistics = DsgStatistics(graph)
statistics.print_summary()
query = DsgQuery(graph)
all_objects = query.get_objects()
print(len(all_objects))

# 可视化 GVD 图
print("\n正在绘制 GVD 图...")
print("提示: 节点的 size 表示 distance（离障碍物的距离）")
print("      边的 weight 表示 Clearance（安全宽度），不是距离")
print("      如果 edge.weight < robot_radius，该边会被过滤掉（太窄，机器人无法通过）")

# 获取 place 节点和边
places = query.get_places()
edges = query.get_place_edges()
print(f"\n找到 {len(places)} 个 Place 节点，{len(edges)} 条边")

# 调试：如果边数为0，检查 layer 信息
if len(edges) == 0:
    print("\n调试信息:")
    if graph.has_layer(dsg.DsgLayers.PLACES):
        layer = graph.get_layer(dsg.DsgLayers.PLACES)
        print(f"  PLACES Layer 有 {layer.num_nodes()} 个节点")
        print(f"  PLACES Layer 有 {layer.num_edges()} 条边")
        # 检查第一个节点的 siblings
        for node in layer.nodes:
            try: 
                siblings = list(node.siblings())
                print(f"  节点 {node.id.value} 有 {len(siblings)} 个邻居")
                if len(siblings) > 0:
                    print(f"    第一个邻居: {siblings[0]}")
            except Exception as e:
                print(f"  获取节点 {node.id.value} 的 siblings 失败: {e}")
            break

# 可视化 GVD 图
# robot_radius: 机器人半径，用于过滤太窄的边
# min_clearance: 最小 clearance 阈值
visualize_gvd_graph(query, robot_radius=0.3, min_clearance=0.0, 
                    show_spheres=True, show_edges=True)

# print("\n查看所有物体的 name 属性...")
# all_objects = query.get_objects()
# print(f"总共有 {len(all_objects)} 个物体")
# for i, obj in enumerate(all_objects):
#     name_str = obj.name if obj.name else "(None)"
#     print(f"  [{i}] node_id: {obj.node_id}, name: '{name_str}' (type: {type(obj.name)})")
#     if "box" in str(name_str).lower():  # 检查是否包含 box（不区分大小写）
#         print(f"      ^^^ 这个可能是 box 物体")

# # 查询名称为 "box" 的物体（精确匹配）
# print("\n查询 name='box' 的物体（精确匹配）...")
# box_objects = query.get_objects(name="box", node_id = NodeSymbol('O' ,98))

# print(type(box_objects))

# if (isinstance(box_objects, NodeInfo)):
#     box_objects = [box_objects]

# # 如果没找到，尝试不区分大小写查找
# if len(box_objects) == 0:
#     print("\n尝试查找包含 'box' 的物体（不区分大小写）...")
#     for obj in all_objects:
#         if obj.name and "box" in obj.name.lower():
#             print(f"  找到: node_id={obj.node_id}, name='{obj.name}'")
#             box_objects.append(obj)

# # 打印节点 ID
# for obj in box_objects:
#     print(f"\nNode ID: {obj.node_id}, Name: {obj.name}, Position: {obj.position}")
    
#     # 尝试多种方法获取 NodeSymbol 格式
#     try:
#         node_symbol = NodeSymbol(obj.node_id)
        
#         # 方法1: 尝试转换为字符串
#         node_label = str(node_symbol)
#         print(f"  Node Symbol (str): {node_label}")
#     except Exception as e1:
#         print(f"  Method 1 (str) failed: {e1}")


# # 可视化
# print("\n正在启动可视化器...")
# print("提示: 关闭可视化窗口以退出")
# # Set show_node_ids=True to see the node IDs in the visualization
# DsgVisualizer.visualize_static(dsg_file, mesh_file, start_remote=True, show_node_ids=True)

