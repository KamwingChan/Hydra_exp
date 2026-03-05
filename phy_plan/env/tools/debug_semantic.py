from omnigibson.utils.constants import semantic_class_id_to_name

# 检查这些 ID 是否在映射中
id_to_name = semantic_class_id_to_name()
invalid_ids = [12786, 65302]

for invalid_id in invalid_ids:
    if invalid_id in id_to_name:
        print(f"ID {invalid_id} 对应类别: {id_to_name[invalid_id]}")
    else:
        print(f"ID {invalid_id} 不在标准映射中，可能是未映射的像素")