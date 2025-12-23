import json
import os
from omnigibson.utils.constants import semantic_class_id_to_name

def export_only_id_to_name(filename="omnigibson_id_to_name.json"):
    print("正在获取 ID -> Name 映射...")
    
    # 获取原始字典 {int: str}
    # 例如: {402738670: "floor", ...}
    id_to_name_map = semantic_class_id_to_name()
    
    # 写入文件
    try:
        with open(filename, 'w') as f:
            # indent=4 让文件可读性更好
            # sort_keys=True 让 ID 从小到大排序，方便查找
            json.dump(id_to_name_map, f, indent=4, sort_keys=True)
            
        print(f"✅ 成功！文件已保存: {os.path.abspath(filename)}")
        print(f"   包含 {len(id_to_name_map)} 个类别。")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")

if __name__ == "__main__":
    export_only_id_to_name()