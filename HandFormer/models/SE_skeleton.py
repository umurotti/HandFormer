from abc import ABC, abstractmethod
import torch
import lietorch

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
    