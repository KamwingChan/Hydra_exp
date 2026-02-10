"""
ID mapping utility for semantic segmentation.
Maps large IDs to small IDs for compatibility with downstream systems.
"""
import rospy
import numpy as np
import yaml
from pathlib import Path


class IDMapper:
    """Handles semantic ID mapping (large ID -> small ID)."""
    
    def __init__(self, remap_file_path=None):
        """
        Initialize ID mapper.
        
        Args:
            remap_file_path: Path to behavior_remap.yaml file. If None, uses default location.
        """
        self.id_mapping = {}  # 大ID -> 小ID 的字典
        self._load_id_mapping(remap_file_path)
    
    def _load_id_mapping(self, remap_file_path=None):
        """
        加载 ID 映射表（大ID -> 小ID）
        从 env/behavior_remap.yaml 加载
        """
        if remap_file_path is None:
            # 获取当前文件所在目录的父目录（env/）
            env_dir = Path(__file__).parent.parent
            remap_file_path = env_dir / "config" / "label_remaps" / "behavior.yaml"
        
        if not Path(remap_file_path).exists():
            rospy.logerr(f"Remap file not found: {remap_file_path}")
            rospy.logerr("Please copy behavior.yaml to env/config/label_remaps/behavior.yaml")
            # 至少映射 0 -> 0 (unknown)，避免完全崩溃
            self.id_mapping[0] = 0
            return
        
        try:
            with open(remap_file_path, 'r') as f:
                remap_data = yaml.safe_load(f)
                for item in remap_data:
                    big_id = item['sub_id']
                    small_id = item['super_id']
                    self.id_mapping[big_id] = small_id
            rospy.loginfo(f"Loaded {len(self.id_mapping)} ID mappings from: {remap_file_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load remap yaml: {e}")
            # 至少映射 0 -> 0 (unknown)，避免完全崩溃
            self.id_mapping[0] = 0
    
    def remap(self, sem_data):
        """
        将语义图像中的大ID映射为小ID
        优化：使用 np.unique + inverse 避免多次遍历全图
        
        Args:
            sem_data: numpy array of semantic segmentation data (uint32 or uint16)
            
        Returns:
            remapped: numpy array with mapped IDs (uint16)
        """
        if not self.id_mapping:
            return sem_data.astype(np.uint16)
        
        # 转换为 uint32 确保能容纳大ID
        sem_uint32 = sem_data.astype(np.uint32)
        
        # 1. 获取唯一值和反向索引
        # unique_ids: 图像中出现的大ID (sorted)
        # inverse: 原始图像拍平后，每个像素对应在 unique_ids 中的下标
        unique_ids, inverse = np.unique(sem_uint32, return_inverse=True)
        
        # 2. 向量化查找：只对图像中出现的少量唯一ID进行字典查找
        # 相比对全图每个像素查找，或者对每个ID做全图掩码，效率极高
        mapped_vals = np.array([self.id_mapping.get(int(uid), 0) for uid in unique_ids], dtype=np.uint16)
        
        # 3. 重构图像：利用 numpy 高级索引直接生成结果
        # mapped_vals[inverse] 会根据索引一次性生成新的像素值数组
        remapped = mapped_vals[inverse].reshape(sem_data.shape)
        
        return remapped
