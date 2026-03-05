"""
Utilities for using custom traversability maps with OmniGibson.

Our generated map (generate_trav_map_accurate.py) is compatible with
OmniGibson TraversableMap: square, center=world(0,0), 255=traversable, 0=obstacle.
"""
import os

import cv2
import torch as th


def inject_custom_trav_map(env, custom_map_path, custom_resolution=0.01):
    """
    Use custom trav map for navigation (get_shortest_path etc.). When custom_resolution=0.01 the map is kept at full res (no wall loss).
    Compatible with OmniGibson TraversableMap: square, center=world(0,0), 255=traversable, 0=obstacle.

    Args:
        env: OmniGibson Environment (already created)
        custom_map_path: path to our generated floor_trav_0.png
        custom_resolution: map resolution (m/px). 0.01 matches generate_trav_map_accurate.py default.
    """
    if not os.path.isfile(custom_map_path):
        raise FileNotFoundError(f"Custom trav map not found: {custom_map_path}")

    scene = env.scene
    trav = scene._trav_map
    img = cv2.imread(custom_map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {custom_map_path}")

    h, w = img.shape
    if h != w:
        raise ValueError(f"Custom trav map must be square, got {w}x{h}")

    img = (img >= 255).astype("uint8") * 255

    # 注入时一律以自定义图为准，保证导航用的是这张图的分辨率（不被 layout 的 map_size 缩小）
    trav.map_resolution = custom_resolution
    trav.trav_map_original_size = h
    map_size = int(h * 0.01 / custom_resolution)  # PNG 按 0.01 m/px 存，custom_resolution=0.01 时 map_size=h
    if map_size != h:
        img = cv2.resize(img, (map_size, map_size))
    img[img < 255] = 0

    if trav.floor_map is None or len(trav.floor_map) == 0:
        trav.floor_map = [th.tensor(img.copy())]
    else:
        trav.floor_map[0] = th.tensor(img.copy())
    trav.map_size = map_size
    if trav.floor_heights is None:
        trav.floor_heights = (0.0,)

    # 调试：确认导航将使用的分辨率与尺寸
    sh = trav.floor_map[0].shape
    print(f"[TravMap] Injected: map_resolution={trav.map_resolution} m/px, map_size={trav.map_size}, floor_map[0].shape={sh}")
