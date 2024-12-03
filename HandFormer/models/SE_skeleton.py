import sys

from abc import ABC, abstractmethod
import os
import torch
import lietorch
import yaml

# Add the directory containing `feeders` to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feeders.feeder import Feeder

class Skeleton(ABC):
    def __init__(self, hand_pose: torch.Tensor) -> None:
        # hand_pose: (batch_size, 2, J)
        self.hand_pose = hand_pose
        self.vertices = None
        self.edges = None
        self.use_head = False
        pass
    
    def get_edges(self):
        return self.edges
    
    def get_vertices(self):
        return self.vertices
    
    def get_hand_pose(self):
        return self.hand_pose
    
    @abstractmethod
    def calculate_C_t():
        pass
    
    @abstractmethod
    def calculate_c_t():
        pass
    
if __name__ == '__main__':
    main_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.abspath(os.path.join(main_dir, "../train_config.yaml"))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = {}
    data_loader['train'] = torch.utils.data.DataLoader(
    dataset= Feeder(**(config["train_feeder_args"])),
    batch_size=5,
    shuffle=True,
    num_workers=8,
    drop_last=True,)
    
    for i, data in enumerate(data_loader['train']):
        print(data[0].shape) # Tensor: [batch_size, 3, sample_cnt_pose=120, joints=21, no_of_hands=2]
        print(data[1].shape) # Tensor: [batch_size, sample_cnt_vid=8, rgb_feature_dim=1536]
        break