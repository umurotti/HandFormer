import sys

from abc import ABC, abstractmethod
import os
import torch
from lietorch import SE3, SO3
from pytorch3d import transforms
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
    
    def align_vectors_with_rotation(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the rotation matrices that align source points to target points for a batch of vectors,
        without using loops.
        
        Args:
            source (torch.Tensor): Source points as a tensor of shape (n, 3).
            target (torch.Tensor): Target points as a tensor of shape (n, 3).
            
        Returns:
            torch.Tensor: Rotation matrices of shape (n, 3, 3).
        """
        # Normalize the points to treat them as vectors
        source = source / torch.linalg.norm(source, dim=1, keepdim=True)
        target = target / torch.linalg.norm(target, dim=1, keepdim=True)
        
        # Compute the cross product (rotation axis) and dot product (cosine of angle)
        v = torch.cross(source, target, dim=1)  # (n, 3)
        c = torch.sum(source * target, dim=1, keepdim=True)  # (n, 1), Cosine of the angle
        s = torch.linalg.norm(v, dim=1, keepdim=True)       # (n, 1), Sine of the angle

        # Normalize the rotation axis
        v = torch.where(s > 1e-6, v / s, v)  # Avoid division by zero

        # Rodrigues' rotation formula components
        vx = torch.zeros((source.size(0), 3, 3), device=source.device)  # (n, 3, 3)
        vx[:, 0, 1] = -v[:, 2]
        vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2]
        vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]
        vx[:, 2, 1] = v[:, 0]

        identity = torch.eye(3, device=source.device).unsqueeze(0)  # (1, 3, 3)
        R = identity + s[:, :, None] * vx + (1 - c)[:, :, None] * torch.bmm(vx, vx)  # (n, 3, 3)

        # Handle parallel and anti-parallel cases
        parallel_mask = (s < 1e-6).squeeze()  # Boolean mask for parallel/anti-parallel vectors
        anti_parallel_mask = parallel_mask & (c.squeeze() < 0)  # Anti-parallel vectors

        if parallel_mask.any():
            # Handle parallel vectors
            R[parallel_mask] = identity.expand(parallel_mask.sum(), -1, -1)

        if anti_parallel_mask.any():
            # Handle anti-parallel vectors
            R[anti_parallel_mask] = -identity.expand(anti_parallel_mask.sum(), -1, -1)

        return R

class SE_Skeleton(Skeleton):
    def __init__(self, vertices: torch.Tensor) -> None:
        # vertices: (batch_size, sample_cnt_pose=120, 2, joints=21, 3)
        super().__init__(vertices)
        
        self.align_axis = torch.asarray([1, 0, 0], dtype=torch.float32)[None, :] # bais-1 for global x-axis
        self.calculate_C_t()
        self.calculate_c_t()
        

    def calculate_C_t(self):
        # edges: (batch_size, sample_cnt_pose=120, no_of_hands=2, no_of_edges=23, start_and_end_positions=2, 3)
        edge_start = self.edges[:, :, :, :, 0, :] # (batch_size, sample_cnt_pose=120, no_of_edges=23, 3)
        edge_end = self.edges[:, :, :, :, 1, :] # (batch_size, sample_cnt_pose=120, no_of_edges=23, 3)
        
        # Calculate the edge vectors
        edge_vectors = edge_end - edge_start # (batch_size, sample_cnt_pose=120, no_of_edges=23, 3)
        
        # Normalize direction vectors to create a local coordinate system
        edge_directions = edge_vectors / torch.norm(edge_vectors, dim=-1, keepdim=True)
        
        pass
    
    def calculate_c_t(self):
        pass
    
    
    '''
    Given two edge vectors, calculate the P_mn ∈ SE(3) and P_nm ∈ SE(3)
    
    '''
    def calculate_P_t(self, e_n1: torch.Tensor, e_n2: torch.Tensor, e_m1: torch.Tensor, e_m2: torch.Tensor) -> transforms.Transform3d:
        # e_n1, e_n2, e_m1, e_m2: (no_of_edges=23, 3)
        assert e_n1.dim() == e_n2.dim() == e_m1.dim() == e_m2.dim() == self.align_axis.dim() == 2
        
                                       
        e_n2 = e_n2 - e_n1 # bone1_end = bone1_end - bone1_st;
        e_m1 = e_m1 - e_n1 # bone2_st = bone2_st - bone1_st;
        e_m2 = e_m2 - e_n1 # bone2_end = bone2_end - bone1_st;
        e_n1 = e_n1 - e_n1 # bone1_st = bone1_st - bone1_st;
        
        # Find the rotation matrix that converts bone1_global into x-axis
        R = self.align_vectors_with_rotation(source=e_n2, target=self.align_axis) # R = vrrotvec2mat(vrrotvec(bone1_end, [1, 0, 0]));
        
        e_n2 = (R @ e_n2[:, :, None]).squeeze(dim=-1) # bone1_end = R*bone1_end;
        e_m1 = (R @ e_m1[:, :, None]).squeeze(dim=-1)  # bone2_st = R*bone2_st;
        e_m2 = (R @ e_m2[:, :, None]).squeeze(dim=-1)  # bone2_end = R*bone2_end;
        
        # Find the rotation matrix that converts bone1 into bone2
        # This rotation matrix gives us the rotation between both bones in a
        # coordinate system that is attached to bone 1.
        R = self.align_vectors_with_rotation(source=(e_n2 - e_n1), target=(e_m2 - e_m1)) # R = vrrotvec2mat(vrrotvec(bone1_end - bone1_st, bone2_end - bone2_st));
        t = e_m1 - e_n1 # t = bone2_st - bone1_st;
        
        # Form the transformation matrix in SE(3)
        G = torch.cat([
                torch.cat([R, t[:, :, None]], dim=-1),
                torch.asarray([0, 0, 0, 1], device=R.device, dtype=R.dtype)[None, None,:].expand(R.shape[0], -1, -1)
            ], dim=1) # G = [R, t; 0, 0, 0, 1];
        G = transforms.Transform3d(matrix=G.transpose(-1, -2), device='cuda') # se3_points{i} = [R t; [0, 0, 0, 1]];
        
        return G
        
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
        # Example usage
        hand = SE_Skeleton(data[0].permute(0, 2, 4, 3, 1))
        hand.calculate_C_t()
        
        break