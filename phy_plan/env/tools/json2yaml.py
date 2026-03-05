import json
import yaml
import re
from pathlib import Path

def generate_name_descriptive(name):
    """
    生成 name_descriptive 字段
    格式参考 ade20k_full_label_space.yaml
    """
    name_lower = name.lower()
    
    # 处理特殊情况
    if name_lower in ['sky']:
        return f"the {name_lower}"
    elif name_lower in ['person', 'animal', 'unknown']:
        return f"a {name_lower}" if name_lower != 'unknown' else name_lower
    elif name_lower.startswith(('a ', 'an ', 'the ')):
        # 如果已经包含冠词，直接使用
        return name_lower
    elif name_lower in ['water', 'grass', 'dirt', 'sand', 'earth', 'land', 'food', 'coffee']:
        # 不可数名词，不加冠词或加 "the"
        return name_lower
    elif name_lower in ['floor', 'ceiling', 'wall', 'ground']:
        return f"a {name_lower}"
    elif name_lower.endswith('s') and not name_lower.endswith(('ss', 'us', 'is', 'as')):
        # 复数形式
        return f"{name_lower}"
    else:
        # 默认加 "a"
        # 检查是否以元音开头
        if name_lower[0] in 'aeiou':
            return f"an {name_lower}"
        else:
            return f"a {name_lower}"

def convert_omnigibson_to_label_space(
    json_file="omnigibson_id_to_name.json",
    output_file="omnigibson_label_space.yaml"
):
    """
    将 omnigibson_id_to_name.json 转换为 behavior_label_space.yaml 格式
    参考 ade20k_full_label_space.yaml，只排除明显的结构/环境元素
    """
    
    # 定义需要排除的结构/环境元素（精确匹配或作为完整单词）
    # 这些是与 scene graph 和操作无关的背景/结构元素
    structural_exact_matches = {
        # 建筑结构
        'wall', 'walls', 'ceiling', 'ceilings', 'floor', 'floors',
        'roof', 'door', 'sliding_door', 'garage_door', 'elevator_door',
        'window', 'fixed_window', 'openable_window', 'windowpane',
        'window_blind', 'shutter', 'gate', 'fence', 'rail_fence',
        'column', 'beam', 'pillar', 'railing', 'bannister',
        'baseboard', 'molding', 'trim', 'base',
        
        # 建筑/环境
        'building', 'structure', 'foundation', 'house', 'skyscraper',
        'tower', 'arcade', 'hovel', 'kitchen', 'bar', 'booth',
        'awning', 'canopy', 'grandstand', 'stage', 'pier',
        
        # 地面/表面（作为表面，不是物体）
        'ground', 'earth', 'dirt', 'sand', 'gravel', 'soil',
        'asphalt', 'concrete', 'ceramic_tile', 'tile', 'paving_stone',
        'paver', 'driveway', 'sidewalk', 'road', 'path', 'runway',
        'field', 'land', 'hill', 'mountain', 'rock', 'boulder',
        'pebble', 'grass', 'mulch', 'compost', 'peat', 'moss',
        'snow', 'ice', 'seawater', 'water', 'river', 'lake',
        'sea', 'bridge', 'waterfall', 'swimming_pool', 'wading_pool',
        'pool', 'fountain',
        
        # 天空/背景
        'sky', 'background', 'object', 'unlabelled',
        
        # 楼梯/台阶（结构性的）
        'stairs', 'stair', 'stairway', 'step', 'escalator',
        
        # 抽象/不可操作的概念
        'debris', 'chaff', 'dust', 'lint', 'scum', 'sludge',
        'foam', 'ash', 'rust', 'tarnish', 'stain', 'patina',
        'mildew', 'mold', 'wrinkle', 'incision', 'grit',
        'adhesive_material', 'glue', 'tape', 'wax', 'paraffin_wax',
        'rope', 'twine', 'string', 'thread', 'cotton_thread',
        'hair', 'feather', 'fur', 'wool', 'lace', 'ribbon',
        'tinsel', 'wreath', 'decoration', 'ornament',
        'money', 'coin', 'penny', 'nickel', 'quarter', 'dime',
        'silver_coin', 'gold_coin', 'diamond', 'ruby', 'pearl',
        'jade', 'quartz', 'crystal', 'geode', 'seashell', 'conch',
        'antlers', 'bone', 'skull', 'skeleton', 'skeletal_frame',
        'plant_stem', 'branch', 'half_branch', 'log', 'half_log',
        'tree', 'low_resolution_tree', 'bush', 'greenery',
        'flower', 'flower_petal', 'dahlia_flower', 'rose', 'tulip',
        'lily', 'lily_pad', 'sunflower', 'half_sunflower',
        'marigold', 'pottable_marigold', 'pottable_dahlia',
        'pottable_daffodil', 'pottable_cactus', 'pottable_beefsteak_tomato',
        'pottable_chili', 'hanging_plant', 'garden_plant', 'plant_pot',
        'plant_pot_stand', 'terrarium', 'pot_plant', 'echeveria_elegans',
        'cactus', 'poinsettia', 'christmas_tree', 'christmas_tree_decorated',
        'christmas_stocking', 'holly_decoration', 'valentine_wreath',
        'icicle_lights', 'fairy_light', 'wind_chime', 'garden_statue',
        'garden_umbrella', 'garden_chair', 'garden_coffee_table',
        'garden_light', 'pergola', 'arbor', 'sandbox', 'playground',
        'swing_set', 'trampoline', 'trampoline_top', 'trampoline_leg',
        'parallel_bars', 'pommel_horse', 'bench_press_machine',
        'elliptical_machine', 'exercise_bike', 'treadmill',
        'free_weight_rack', 'free_weight', 'weight_bar', 'gym_mat',
        'yoga_mat', 'rolled_yoga_mat', 'punching_bag', 'boxing_gloves',
        'batting_gloves', 'goalkeeper_gloves', 'baseball_glove', 'gym_shoe',
        'hiking_boot', 'rubber_boot', 'sandal', 'slingback', 'high_heel',
        'baby_shoe', 'skates', 'skateboard_wheel', 'skateboard_wheel_assembly',
        'skateboard_deck', 'longboard', 'bicycle', 'bicycle_rack',
        'bicycle_chain', 'motorcycle', 'car', 'pickup_truck', 'trailer_truck',
        'recreational_vehicle', 'lorry', 'car_wheel', 'whitewall_tire',
        'license_plate', 'windshield', 'hitch', 'fuel_can', 'gasoline',
        'antifreeze', 'oil', 'bottle_of_oil', 'cooking_oil',
        'cooking_oil_atomizer', 'olive_oil', 'olive_oil_bottle',
        'coconut_oil', 'coconut_oil_jar', 'bottle_of_coconut_oil',
        'canola_oil', 'peanut_oil', 'sesame_oil', 'sesame_oil_bottle',
        'bottle_of_sesame_oil', 'almond_oil', 'bottle_of_almond_oil',
        'linseed_oil', 'baby_oil', 'bottle_of_baby_oil', 'lavender_oil',
        'bottle_of_lavender_oil', 'essential_oil', 'bottle_of_essential_oil',
        'lubricant', 'lubricant_bottle', 'bottle_of_lubricant', 'grease',
        'vaseline', 'polish', 'polish_bottle', 'wax_paper', 'aluminum_foil',
        'box_of_aluminium_foil', 'plastic_wrap', 'cellulose_tape',
        'cellulose_tape_dispenser', 'masking_tape', 'duct_tape',
        'spackle', 'grout', 'caulk', 'bottle_of_caulk', 'sealant',
        'sealant_atomizer', 'bottle_of_sealant', 'paint', 'house_paint',
        'bucket_of_paint', 'bottle_of_paint', 'spray_paint', 'spray_paint_can',
        'paint_roller', 'paintbrush', 'varnish', 'lacquer', 'bottle_of_lacquer',
        'floor_wax', 'floor_wax_bottle', 'wax_remnant', 'paraffin_wax',
        'beeswax_candle', 'dip_candle', 'pillar_candle', 'candle_holder',
        'match', 'match_box', 'lighter', 'butane', 'gunpowder', 'charcoal',
        'charcoal_grill', 'firewood', 'firewood_grate', 'firewood_chopping_block',
        'ember', 'ash', 'coal', 'hay', 'straw', 'mulch', 'mulch_bag',
        'bag_of_mulch', 'fertilizer', 'fertilizer_atomizer', 'bag_of_fertilizer',
        'compost', 'compost_bin', 'soil', 'soil_bag', 'dirt', 'sand', 'gravel',
        'heap_of_gravel', 'pebble', 'rock', 'boulder', 'stone_wall', 'concrete',
        'asphalt', 'ceramic_tile', 'tile', 'paving_stone', 'paver', 'driveway',
        'sidewalk', 'road', 'bridge', 'slope', 'stairs', 'stair', 'railing',
        'baseboard', 'molding', 'trim', 'column', 'beam', 'pillar', 'pedestal',
        'signpost', 'flagpole', 'flag_pole', 'pennant', 'banner', 'hanging_banners',
        'sign', 'decorative_sign', 'information_bulletin', 'bulletin_board',
        'price_tag', 'name_tag', 'tag', 'label', 'sticker', 'postage_stamp',
        'postcard', 'letter', 'envelope', 'package', 'mail', 'legal_document',
        'architectural_plan', 'map', 'globe', 'periodic_table', 'chart', 'menu',
        'receipt', 'ticket', 'address', 'nectar', 'sap', 'weed',
    }
    
    # 定义动态标签：只有明确的动态物体（会移动的活体）
    # 参考 ade20k_full_label_space.yaml，只有 person 和 animal 是动态的
    # 使用精确匹配或明确的动态物体名称
    dynamic_exact_matches = {
        'agent',           # 智能体/机器人
        'person',         # 人
        'human',          # 人类
        'people',         # 人们
        'man',            # 男人
        'woman',          # 女人
        'child',          # 孩子
        'baby',           # 婴儿
        'animal',         # 动物（活的）
    }
    
    # 动态物体关键词（必须是完整单词，且不是食物）
    dynamic_keywords = {
        'person', 'human', 'people', 'man', 'woman', 'child', 'baby',
        'animal', 'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'pig',
        'sheep', 'goat', 'rabbit', 'mouse', 'rat',
        'hamster', 'guinea_pig', 'turtle', 'snake', 'lizard', 'frog',
        'insect', 'spider', 'bee', 'butterfly', 'fly', 'mosquito',
    }
    
    # 排除列表：包含动态关键词但不是动态物体的名称模式
    dynamic_exclude_patterns = [
        r'cooked__.*',      # 烹饪过的食物
        r'diced__.*',       # 切碎的食物
        r'half_.*',         # 一半的食物
        r'.*_leg',          # 腿（如 chicken_leg）
        r'.*_breast',       # 胸（如 chicken_breast）
        r'.*_wing',         # 翅膀（如 chicken_wing）
        r'.*_tender',       # 嫩肉
        r'.*_seasoning',    # 调料
        r'.*_powder',       # 粉末
        r'.*_jar',          # 罐子
        r'.*_bottle',       # 瓶子
        r'.*_can',          # 罐头
        r'.*_box',          # 盒子
        r'.*_package',      # 包装
        r'.*_bag',          # 袋子
        r'.*_carton',       # 纸盒
        r'.*_frank',        # 法兰克福香肠（如 hotdog_frank）
        r'.*hotdog.*',      # 热狗相关
        r'perspiration',    # 汗水
        r'pants',           # 裤子
        r'mannequin',       # 人体模型
        r'beet',            # 甜菜（包含"bee"但不是蜜蜂）
        r'.*beet.*',        # 甜菜相关
        r'.*duck.*',        # 鸭肉相关（食物）
        r'.*chicken.*',     # 鸡肉相关（食物）
        r'.*turkey.*',      # 火鸡肉相关（食物）
        r'.*fish.*',        # 鱼肉相关（食物）
        r'.*beef.*',        # 牛肉相关（食物）
        r'.*pork.*',        # 猪肉相关（食物）
        r'.*lamb.*',        # 羊肉相关（食物）
        r'.*veal.*',        # 小牛肉相关（食物）
        r'^mouse$',         # 计算机鼠标（单独出现时，不是动物）
        r'computer.*mouse', # 计算机鼠标
        r'.*mouse.*pad',    # 鼠标垫
        r'.*mouse.*trap',   # 捕鼠器（不是动物本身）
    ]
    
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        id_to_name = json.load(f)
    
    # 将字符串键转换为整数
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    
    # 按 ID 排序
    sorted_items = sorted(id_to_name.items())
    
    # 分类：对象 vs 结构元素
    objects = []
    structural = []
    
    for obj_id, obj_name in sorted_items:
        # 精确匹配检查
        if obj_name in structural_exact_matches:
            structural.append((obj_id, obj_name))
        else:
            # 保留所有其他对象（包括家具、可操作物品等）
            objects.append((obj_id, obj_name))
    
    # 生成 label 映射
    # label 0 保留给 Unknown，其他使用原始ID
    label_names = []
    dynamic_labels = []
    object_labels = []
    
    # 首先添加 Unknown (label 0)
    label_names.append({
        "label": 0, 
        "name": "Unknown",
        "name_descriptive": "unknown"
    })
    
    # 先添加结构元素（不包含在 object_labels 中），使用原始ID
    for obj_id, obj_name in structural:
        name_descriptive = generate_name_descriptive(obj_name)
        label_names.append({
            "label": obj_id, 
            "name": obj_name,
            "name_descriptive": name_descriptive
        })
    
    # 再添加对象（包含在 object_labels 中），使用原始ID
    for obj_id, obj_name in objects:
        # 检查是否是动态标签
        obj_name_lower = obj_name.lower()
        is_dynamic = False
        
        # 先检查精确匹配
        if obj_name_lower in dynamic_exact_matches:
            is_dynamic = True
        else:
            # 先检查排除模式
            excluded = False
            for pattern in dynamic_exclude_patterns:
                if re.match(pattern, obj_name_lower):
                    excluded = True
                    break
            
            if not excluded:
                # 使用单词边界匹配关键词（确保是完整单词）
                for keyword in dynamic_keywords:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, obj_name_lower):
                        is_dynamic = True
                        break
        
        if is_dynamic:
            dynamic_labels.append(obj_id)
        
        object_labels.append(obj_id)
        name_descriptive = generate_name_descriptive(obj_name)
        label_names.append({
            "label": obj_id, 
            "name": obj_name,
            "name_descriptive": name_descriptive
        })
    
    # 计算 total_semantic_labels（最大ID + 1）
    all_ids = [0] + [item["label"] for item in label_names if item["label"] != 0]
    total_semantic_labels = max(all_ids) + 1 if all_ids else 1
    
    # 识别 surface_places_labels（可行走的地面/平面区域，用于 Place2D 分割）
    # 参考 ade20k_full_label_space.yaml: floor, road, grass, sidewalk, earth, water, 
    # field, sand, path, stairs, runway, stairway, river, bridge, hill, dirt, land, stage, pier
    # 注意：不包括 table, desk, counter, shelf 等家具，这些虽然可以放置物品，
    # 但不是"可行走的地面"，不应该用于 Place2D 分割
    surface_places = []
    surface_keywords = [
        'floor', 'stair', 'stairway', 'ground', 'path', 'road', 
        'sidewalk', 'field', 'land', 'hill', 'sand', 'dirt',
        'earth', 'grass', 'water', 'bridge', 'runway', 'stage',
        'pier', 'river', 'lake', 'sea', 'escalator', 'ramp',
        'platform', 'terrace', 'patio', 'deck', 'walkway', 'corridor',
        'hallway', 'passage', 'aisle', 'slope', 'trail', 'track'
    ]
    
    # 排除列表：包含表面关键词但不是表面的物体
    surface_exclude_patterns = [
        r'.*sandal.*',      # 凉鞋（包含"sand"但不是沙子）
        r'.*sandbox.*',     # 沙盒（虽然是表面，但通常不作为可行走区域）
    ]
    
    for obj_id, obj_name in structural:
        obj_name_lower = obj_name.lower()
        
        # 先检查排除模式
        excluded = False
        for pattern in surface_exclude_patterns:
            if re.match(pattern, obj_name_lower):
                excluded = True
                break
        
        if not excluded:
            # 使用单词边界匹配，确保是完整单词匹配
            for kw in surface_keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, obj_name_lower):
                    surface_places.append(obj_id)
                    break
    
    # 写入 YAML 文件，手动控制格式以确保顺序和格式正确
    with open(output_file, 'w') as f:
        f.write("---\n")
        f.write(f"total_semantic_labels: {total_semantic_labels}\n")
        
        # dynamic_labels
        dynamic_labels_sorted = sorted(dynamic_labels) if dynamic_labels else []
        if dynamic_labels_sorted:
            f.write(f"dynamic_labels: {dynamic_labels_sorted}\n")
        else:
            f.write("dynamic_labels: []\n")
        
        # invalid_labels
        f.write("invalid_labels: [0]\n")
        
        # object_labels
        object_labels_sorted = sorted(object_labels)
        f.write("object_labels:\n")
        for label in object_labels_sorted:
            f.write(f"  - {label}\n")
        
        # surface_places_labels (在 object_labels 之后)
        if surface_places:
            surface_places_sorted = sorted(surface_places)
            f.write("surface_places_labels:\n")
            for label in surface_places_sorted:
                f.write(f"  - {label}\n")
        
        # label_names (在 surface_places_labels 之后)
        # 按 label ID 排序
        label_names_sorted = sorted(label_names, key=lambda x: x['label'])
        f.write("label_names:\n")
        for item in label_names_sorted:
            # 使用 YAML 格式：- {label: 0, name: wall, name_descriptive: a wall}
            # 注意：双花括号 {{ }} 用于转义，生成单个花括号
            f.write(f"  - {{label: {item['label']}, name: {item['name']}, name_descriptive: {item['name_descriptive']}}}\n")
    
    print(f"✅ 转换完成！")
    print(f"   输入文件: {json_file}")
    print(f"   输出文件: {output_file}")
    print(f"   总标签数: {total_semantic_labels}")
    print(f"   对象标签数: {len(object_labels)}")
    print(f"   结构元素数: {len(structural)}")
    print(f"   动态标签数: {len(dynamic_labels)}")
    if surface_places:
        print(f"   表面/位置标签数: {len(surface_places)}")
    
    # 打印一些示例
    print(f"\n示例对象标签（前10个）:")
    for i, (obj_id, obj_name) in enumerate(objects[:10]):
        is_dyn = " [动态]" if obj_id in dynamic_labels else ""
        print(f"   {obj_id}: {obj_name}{is_dyn}")
    
    print(f"\n示例结构元素（前10个）:")
    for i, (obj_id, obj_name) in enumerate(structural[:10]):
        print(f"   {obj_id}: {obj_name}")
    
    if dynamic_labels:
        print(f"\n动态标签:")
        for label in sorted(dynamic_labels):
            for item in label_names:
                if item["label"] == label:
                    print(f"   {label}: {item['name']}")
                    break
    
    # 返回生成的数据（用于测试或进一步处理）
    return {
        "total_semantic_labels": total_semantic_labels,
        "dynamic_labels": sorted(dynamic_labels) if dynamic_labels else [],
        "invalid_labels": [0],
        "object_labels": sorted(object_labels),
        "surface_places_labels": sorted(surface_places) if surface_places else [],
        "label_names": label_names
    }

if __name__ == "__main__":
    # 设置文件路径
    json_file = Path(__file__).parent.parent / "omnigibson_id_to_name.json"
    output_file = Path(__file__).parent.parent / "omnigibson_label_space.yaml"
    
    convert_omnigibson_to_label_space(str(json_file), str(output_file))