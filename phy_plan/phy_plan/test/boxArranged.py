from graphReader import DsgVisualizer, DsgStatistics, load_dsg, DsgQuery, NodeInfo
from arrangePolicy import arrange_objects_with_path_planning
import numpy as np
import spark_dsg as dsg
from spark_dsg import NodeSymbol


def main():
    dsg_file = "backend/dsg.json"
    mesh_file = "backend/mesh.ply"

    print("\n[1/4] Loading the graph...")
    graph = load_dsg(dsg_file, mesh_file)
    query = DsgQuery(graph)
    print("Graph loaded successfully!")
    
    print("\n[2/4] Querying the target object...")
    target_node_id = NodeSymbol('O', 98)
    
    box_object = query.get_objects(node_id=target_node_id)
    print(box_object.bounding_box)
    
    # 处理查询结果
    if isinstance(box_object, NodeInfo):
        box_object = [box_object]
    
    if len(box_object) == 0:
        print(f"No object found with node_id={target_node_id}.")
        return
    obj = box_object[0]

    print(f"Found target object:")
    print(f"  - node_id: {obj.node_id}")
    print(f"  - name: {obj.name}")
    print(f"  - current position: {obj.position}")
    
    print("\n[3/4] Setting the target position...")
    target_position = np.array([25.06, -3.93, 0.2])
    print(f"Target position: {target_position}")
    
    print("\n[4/4] Executing path planning and moving...")
    success, path = arrange_objects_with_path_planning(
        graph, 
        obj, 
        target_position, 
        visualize=True,           
        step_delay=5,          
        resolution=0.2,          
        collision_radius=0.5
    )
    if success:
        print("✓ Moving successful!")
        if path:
            path_length = sum(np.linalg.norm(path[i+1]-path[i]) 
                            for i in range(len(path)-1))
            print(f"  - The path contains {len(path)} points")
            print(f"  - The total length of the path: {path_length:.2f} meters")
    else:
        print("✗ Moving failed")
        print("  - You can try to adjust the resolution or collision_radius parameters")
    
    print("\nFinal scene graph statistics:")
    statistics = DsgStatistics(graph)
    statistics.print_summary()


if __name__ == "__main__":
    main()
