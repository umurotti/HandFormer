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
    def __init__(self, vertices: torch.Tensor) -> None:
        # vertices: (batch_size, sample_cnt_pose=120, 2, joints=21, 3)
        self.vertices = vertices
        # Joint connectivity
        '''
          1    2   3   4
          |    |   |   |
         10   13   16  19 
          |    |   |   |
          9   12   15  18
       0  |    |   |   |
       |  8   11   14  17
        7  \   |   /  /
         \  \  |  /  /
           6 \   /  /
            \ -|- /
               5
        '''
        
        self.connectivity = [
            (5, 6), (6, 7), (7, 0),  # Thumb
            (5, 8), (8, 9), (9, 10), (10, 1),  # Index finger
            (5, 11), (11, 12), (12, 13), (13, 2),  # Middle finger
            (5, 14), (14, 15), (15, 16), (16, 3),  # Ring finger
            (5, 17), (17, 18), (18, 19), (19, 4),  # Pinky finger
            #(5, 20), # Wrist COULD NOT UNDERSTAND
            (6, 8), (8, 11), (11, 14), (14,17)  # Fist line
        ]
        # 5 - wrist joint, 11 - finger-root of middle finger (Assembly101)
        
        # edges: (batch_size, sample_cnt_pose=120, 2, no_of_edges=23, start_and_end_positions=2, 3)
        self.edges = self.vertices[:, :, :, self.connectivity, :]
        self.use_head = False
        pass
    
    
    @abstractmethod
    def calculate_C_t():
        pass
    
    @abstractmethod
    def calculate_c_t():
        pass

class SE_Skeleton(Skeleton):
    def __init__(self, vertices: torch.Tensor) -> None:
        # vertices: (batch_size, sample_cnt_pose=120, 2, joints=21, 3)
        super().__init__(vertices)
        self.calculate_C_t()
        self.calculate_c_t()
        

    def calculate_C_t(self):
        pass
    
    def calculate_c_t(self):
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
        if i == 1 or i == 2 or i == 3 or i == 4:
            continue
        print(data[0].shape) # Pose -> Tensor: [batch_size, 3, sample_cnt_pose=120, joints=21, no_of_hands=2]
        print(data[1].shape) # RGB -> Tensor: [batch_size, sample_cnt_vid=8, rgb_feature_dim=1536]
        #print(data[2]) # label -> Tensor: [batch_size]
        #print(data[3]) # verb label -> Tensor: [batch_size]
        #print(data[4]) # noun label -> Tensor: [batch_size]
        #print(data[5]) # index -> Tensor: [batch_size]
        
        hand = SE_Skeleton(data[0].permute(0, 2, 4, 3, 1))
        break