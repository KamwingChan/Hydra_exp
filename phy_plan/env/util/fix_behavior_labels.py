#!/usr/bin/env python3
"""
脚本功能：
1. 从 behavior.csv 读取所有标签 ID，生成大 ID -> 小 ID 的映射
2. 生成 label remap 文件 (behavior.yaml)
3. 修改 behavior_label_space.yaml，将所有大 ID 替换为小 ID
"""

import csv
import yaml
import re
from pathlib import Path

# 文件路径
CSV_FILE = Path("/home/kamwing/catkin_ws/src/hydra_ros/hydra_ros/config/color/behavior.csv")
LABEL_SPACE_FILE = Path("/home/kamwing/catkin_ws/src/hydra/config/label_spaces/behavior_label_space.yaml")
REMAP_OUTPUT = Path("/home/kamwing/catkin_ws/src/hydra/config/label_remaps/behavior.yaml")
LABEL_SPACE_OUTPUT = Path("/home/kamwing/catkin_ws/src/hydra/config/label_spaces/behavior_label_space_new.yaml")

def read_csv_and_create_mapping():
    """读取 CSV 文件，创建大 ID -> 小 ID 的映射"""
    id_to_small_id = {}
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        small_id = 0
        for row in reader:
            big_id = int(row['id'])
            id_to_small_id[big_id] = small_id
            small_id += 1
    
    print(f"读取了 {len(id_to_small_id)} 个标签映射")
    print(f"ID 范围: {min(id_to_small_id.keys())} -> {min(id_to_small_id.values())} 到 {max(id_to_small_id.keys())} -> {max(id_to_small_id.values())}")
    
    return id_to_small_id

def remap_id_in_yaml_value(value, id_to_small_id):
    """递归替换 YAML 值中的大 ID（备用函数，用于 YAML 解析方式）"""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k == 'label' and isinstance(v, int):
                # 替换 label 字段
                result[k] = id_to_small_id.get(v, v)
            else:
                result[k] = remap_id_in_yaml_value(v, id_to_small_id)
        return result
    elif isinstance(value, list):
        return [remap_id_in_yaml_value(item, id_to_small_id) for item in value]
    elif isinstance(value, int):
        # 如果是整数，检查是否在映射中
        return id_to_small_id.get(value, value)
    else:
        return value

def generate_remap_file(id_to_small_id, unknown_labels=None):
    """生成 label remap YAML 文件"""
    remap_data = []
    
    # 按小 ID 排序
    sorted_items = sorted(id_to_small_id.items(), key=lambda x: x[1])
    
    for big_id, small_id in sorted_items:
        remap_data.append({
            'sub_id': big_id,
            'super_id': small_id
        })
    
    # 添加未知标签映射到 0 (invalid)
    if unknown_labels:
        for unknown_id in unknown_labels:
            remap_data.append({
                'sub_id': unknown_id,
                'super_id': 0  # 映射到 invalid label
            })
        print(f"✓ 添加了 {len(unknown_labels)} 个未知标签映射到 invalid (0)")
    
    # 确保输出目录存在
    REMAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    # 按 sub_id 排序以便查找
    remap_data.sort(key=lambda x: x['sub_id'])
    
    with open(REMAP_OUTPUT, 'w') as f:
        f.write("---\n")  # 添加 YAML 文档头
        for item in remap_data:
            # 使用 flow style 格式: - {sub_id: X, super_id: Y}
            f.write(f"- {{sub_id: {item['sub_id']}, super_id: {item['super_id']}}}\n")
    
    print(f"✓ 生成 remap 文件: {REMAP_OUTPUT}")
    return remap_data

def update_label_space_yaml(id_to_small_id):
    """修改 behavior_label_space.yaml，替换所有大 ID，保留原始格式"""
    with open(LABEL_SPACE_FILE, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    i = 0
    replaced_counts = {
        'dynamic_labels': 0,
        'invalid_labels': 0,
        'object_labels': 0,
        'surface_places_labels': 0,
        'ground_labels': 0,
        'label_names': 0
    }
    
    while i < len(lines):
        line = lines[i]
        
        # 处理 dynamic_labels: [ID] 格式
        if line.strip().startswith('dynamic_labels:'):
            # 检查当前行是否已经包含 [ID] 格式
            if '[' in line:
                # 在同一行，直接替换
                ids = re.findall(r'\d+', line)
                if ids:
                    new_ids = [str(id_to_small_id.get(int(id), int(id))) for id in ids]
                    # 替换当前行中的 [ID]
                    new_line = re.sub(r'\[.*?\]', '[' + ', '.join(new_ids) + ']', line)
                    output_lines.append(new_line)
                    replaced_counts['dynamic_labels'] = len(ids)
                else:
                    output_lines.append(line)
                i += 1
            else:
                # 在下一行
                output_lines.append(line)  # 保留原行
                i += 1
                if i < len(lines):
                    content = lines[i]
                    ids = re.findall(r'\d+', content)
                    if ids:
                        new_ids = [str(id_to_small_id.get(int(id), int(id))) for id in ids]
                        new_content = re.sub(r'\[.*?\]', '[' + ', '.join(new_ids) + ']', content)
                        output_lines.append(new_content)
                        replaced_counts['dynamic_labels'] = len(ids)
                    else:
                        output_lines.append(content)
                    i += 1
            continue
        
        # 处理 invalid_labels: [ID] 格式
        if line.strip().startswith('invalid_labels:'):
            # 检查当前行是否已经包含 [ID] 格式
            if '[' in line:
                ids = re.findall(r'\d+', line)
                if ids:
                    new_ids = [str(id_to_small_id.get(int(id), int(id))) for id in ids]
                    new_line = re.sub(r'\[.*?\]', '[' + ', '.join(new_ids) + ']', line)
                    output_lines.append(new_line)
                    replaced_counts['invalid_labels'] = len(ids)
                else:
                    output_lines.append(line)
                i += 1
            else:
                output_lines.append(line)
                i += 1
                if i < len(lines):
                    content = lines[i]
                    ids = re.findall(r'\d+', content)
                    if ids:
                        new_ids = [str(id_to_small_id.get(int(id), int(id))) for id in ids]
                        new_content = re.sub(r'\[.*?\]', '[' + ', '.join(new_ids) + ']', content)
                        output_lines.append(new_content)
                        replaced_counts['invalid_labels'] = len(ids)
                    else:
                        output_lines.append(content)
                    i += 1
            continue
        
        # 处理 object_labels: 块格式列表
        if line.strip().startswith('object_labels:'):
            output_lines.append(line)
            i += 1
            # 读取所有 - ID 行
            count = 0
            while i < len(lines) and lines[i].strip().startswith('- '):
                old_line = lines[i]
                # 提取 ID
                match = re.search(r'- (\d+)', old_line)
                if match:
                    old_id = int(match.group(1))
                    new_id = id_to_small_id.get(old_id, old_id)
                    # 替换 ID，保留格式
                    new_line = re.sub(r'- \d+', f'- {new_id}', old_line)
                    output_lines.append(new_line)
                    count += 1
                else:
                    output_lines.append(old_line)
                i += 1
            replaced_counts['object_labels'] = count
            continue
        
        # 处理 surface_places_labels
        if line.strip().startswith('surface_places_labels:'):
            output_lines.append(line)
            i += 1
            count = 0
            while i < len(lines) and lines[i].strip().startswith('- '):
                old_line = lines[i]
                match = re.search(r'- (\d+)', old_line)
                if match:
                    old_id = int(match.group(1))
                    new_id = id_to_small_id.get(old_id, old_id)
                    new_line = re.sub(r'- \d+', f'- {new_id}', old_line)
                    output_lines.append(new_line)
                    count += 1
                else:
                    output_lines.append(old_line)
                i += 1
            replaced_counts['surface_places_labels'] = count
            continue
        
        # 处理 ground_labels
        if line.strip().startswith('ground_labels:'):
            output_lines.append(line)
            i += 1
            count = 0
            while i < len(lines) and lines[i].strip().startswith('- '):
                old_line = lines[i]
                match = re.search(r'- (\d+)', old_line)
                if match:
                    old_id = int(match.group(1))
                    new_id = id_to_small_id.get(old_id, old_id)
                    new_line = re.sub(r'- \d+', f'- {new_id}', old_line)
                    output_lines.append(new_line)
                    count += 1
                else:
                    output_lines.append(old_line)
                i += 1
            replaced_counts['ground_labels'] = count
            continue
        
        # 处理 label_names: flow style 列表
        if line.strip().startswith('label_names:'):
            output_lines.append(line)
            i += 1
            count = 0
            while i < len(lines):
                old_line = lines[i]
                # 匹配 {label: ID, name: ..., name_descriptive: ...}
                match = re.search(r'\{label: (\d+)', old_line)
                if match:
                    old_id = int(match.group(1))
                    new_id = id_to_small_id.get(old_id, old_id)
                    # 替换 label 值，保留其他格式
                    new_line = re.sub(r'label: \d+', f'label: {new_id}', old_line)
                    output_lines.append(new_line)
                    count += 1
                else:
                    # 如果不是 label_names 的项，可能是下一个 section
                    if old_line.strip() and not old_line.strip().startswith('- '):
                        break
                    output_lines.append(old_line)
                i += 1
            replaced_counts['label_names'] = count
            continue
        
        # 其他行直接保留
        output_lines.append(line)
        i += 1
    
    # 写回文件
    with open(LABEL_SPACE_OUTPUT, 'w') as f:
        f.writelines(output_lines)
    
    # 打印统计信息
    for key, count in replaced_counts.items():
        if count > 0:
            print(f"✓ 替换了 {count} 个 {key}")
    
    print(f"✓ 更新了 label_space 文件: {LABEL_SPACE_OUTPUT} (保留原始格式)")

def main():
    print("=" * 60)
    print("Behavior 标签 ID 映射脚本")
    print("=" * 60)
    
    # 1. 读取 CSV 并创建映射
    print("\n[1/4] 读取 CSV 文件并创建 ID 映射...")
    id_to_small_id = read_csv_and_create_mapping()
    
    # 2. 检查已知的未知标签（从错误日志中发现的）
    known_unknown_labels = [12786, 65302, 36224, 7879]
    print(f"\n[2/4] 检查已知的未知标签: {known_unknown_labels}")
    actual_unknown = [label for label in known_unknown_labels if label not in id_to_small_id]
    if actual_unknown:
        print(f"⚠️  发现 {len(actual_unknown)} 个未知标签: {actual_unknown}")
        print(f"   这些标签将映射到 invalid (0)")
    else:
        print("✓ 所有已知标签都在 CSV 中")
    
    # 3. 生成 remap 文件（包含未知标签映射）
    print("\n[3/4] 生成 label remap 文件...")
    generate_remap_file(id_to_small_id, unknown_labels=actual_unknown if actual_unknown else None)
    
    # 4. 更新 label_space yaml
    print("\n[4/4] 更新 behavior_label_space.yaml...")
    update_label_space_yaml(id_to_small_id)
    
    print("\n" + "=" * 60)
    print("✓ 完成！")
    print("=" * 60)
    if actual_unknown:
        print(f"\n⚠️  注意: {len(actual_unknown)} 个未知标签已映射到 invalid (0)")
        print(f"   如果运行时还出现其他未知标签，请将它们添加到 known_unknown_labels 列表中")
    print(f"\n生成的文件：")
    print(f"  - {REMAP_OUTPUT}")
    print(f"  - {LABEL_SPACE_OUTPUT} (新文件，保留原始格式)")
    print(f"\n下一步：")
    print(f"  1. 检查生成的文件是否正确")
    print(f"  2. 如果正确，可以替换原文件或手动合并")
    print(f"  3. 在 behavior.launch 中添加：")
    print(f'     <arg name="semantic_label_remap_filepath" default="$(find hydra)/config/label_remaps/behavior.yaml"/>')

if __name__ == "__main__":
    main()