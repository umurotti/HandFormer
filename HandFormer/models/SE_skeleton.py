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

class My3DTransform(transforms.Transform3d):
    def __init__(self, matrix: torch.Tensor = None, R: torch.Tensor = None, t: torch.Tensor = None) -> None:
        if matrix is not None:
            # Matrix should be batch of 4x4 SE(3) pytorch matrix
            assert isinstance(matrix, torch.Tensor) and matrix.shape[-2:] == (4, 4) # matrix: (batch_size, 4, 4)
            # Assert column base representation for transformation matrix
            assert torch.allclose(matrix[:, 3, :], torch.tensor([0, 0, 0, 1], device=matrix.device, dtype=matrix.dtype)) # Last row should be [0, 0, 0, 1]
        elif R is not None and t is not None:
            # R: (batch_size, 3, 3), t: (batch_size, 3)
            assert R.dim() == 3 and R.shape[-2:] == (3, 3) # R: (batch_size, 3, 3)
            assert t.dim() == 2 and t.shape[-1] == 3 # t: (batch_size, 3)
            matrix = torch.cat([
                torch.cat([R, t[:, :, None]], dim=-1),
                torch.tensor([0, 0, 0, 1], device=R.device, dtype=R.dtype)[None, None,:].expand(R.shape[0], -1, -1)
            ], dim=1)
        else:
            raise ValueError("Either matrix or R and t should be provided")
        
        '''
        CONVENTIONS of Pytorch3D:
        We adopt a right-hand coordinate system, meaning that rotation about an axis with a positive angle results in a counter clockwise rotation.

        This class assumes that transformations are applied on inputs which are row vectors. The internal representation of the Nx4x4 transformation matrix is of the form:

        M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]

        To apply the transformation to points, which are row vectors, the latter are converted to homogeneous (4D) coordinates and right-multiplied by the M matrix:
        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        [transformed_points, 1] ∝ [points, 1] @ M

        '''
        # We stay in column base representation
        '''
        M = [
                [Rxx,   Rxy,    Rxz,    Tx],
                [Ryx,   Ryy,    Ryz,    Ty],
                [Rzx,   Rzy,    Rzz,    Tz],
                [0,     0,      0,      1],
            ]
        '''
        # Transpose the matrix to convert it to row-base representation used by PyTorch3D
        super().__init__(matrix=matrix.transpose(-1, -2), device=matrix.device, dtype=matrix.dtype)

    def batch_transform_points(self, points: torch.Tensor) -> torch.Tensor:
        # points: (no_of_points, 3)
        assert points.dim() == 2 and points.shape[1] == 3
        
        return torch.bmm(
                        super().get_matrix().transpose(-1, -2)[:, :3, :4], 
                        torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1).unsqueeze(-1)
                        ).squeeze(-1) # e_n2_transformed = G*e_n2;
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
        Computes the rotation matrices that align source points to target points for a batch of vectors.
        
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
        
        self.align_axis = torch.asarray([1, 0, 0], dtype=torch.float32, device=vertices.device)[None, :] # bais-1 for global x-axis
        self.calculate_C_t()
        self.calculate_c_t()
        

    def calculate_C_t(self):
        # edges: (batch_size, sample_cnt_pose=120, no_of_hands=2, no_of_edges=23, start_and_end_positions=2, 3)
        batch_size, sample_cnt_pose, no_of_hands, no_of_edges, start_and_end_positions, _ = self.edges.shape
        edge_start = self.edges.reshape(batch_size, sample_cnt_pose, no_of_hands * no_of_edges, start_and_end_positions, -1)[:, :, :, 0, :] # (batch_size, sample_cnt_pose=120, no_of_hands*no_of_edges=23, 3)
        edge_end = self.edges.reshape(batch_size, sample_cnt_pose, no_of_hands * no_of_edges, start_and_end_positions, -1)[:, :, :, 1, :] # (batch_size, sample_cnt_pose=120, no_of_hands*no_of_edges=23, 3)
        out = []
        for batch in range(batch_size):
            C_t = []
            for sample in range(sample_cnt_pose):
                no_total_edges = no_of_hands * no_of_edges # no_total_edges: [0, 1, 2, ..., 46]
                pairs_n_m = torch.combinations(torch.arange(no_total_edges), 2) # pairs_n_m: [[0, 1], [0, 2], ..., [45, 46]]
                pairs_m_n = torch.combinations(torch.arange(no_total_edges), 2)[:, [1, 0]] # pairs_n_m: [[1, 0], [2, 0], ..., [46, 45]]
                pairs = torch.cat((pairs_n_m, pairs_m_n), dim = 0) # pairs: [[0, 1], [0, 2], ..., [45, 46], [1, 0], [2, 0], ..., [46, 45]]
                if __debug__:
                    # Number of pairs should be equal to the C(no_total_edges, 2) * 2 = no_total_edges * (no_total_edges - 1)
                    assert pairs.shape == (no_total_edges * (no_total_edges - 1), 2)
                # pairs[:, 0]: n
                e_n1 = edge_start[batch, sample, pairs[:, 0]]
                e_n2 = edge_end[batch, sample, pairs[:, 0]]
                # pairs[:, 1]: m
                e_m1 = edge_start[batch, sample, pairs[:, 1]]
                e_m2 = edge_end[batch, sample, pairs[:, 1]]
                
                # [0]-> Transform3d object, [1]-> Tensor: (no. of pairs, 4, 4)
                C_t.append(self.calculate_P_t(e_n1, e_n2, e_m1, e_m2)[1])
                
            out.append(torch.stack(C_t))
        
        return torch.stack(out)
    
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
        G = My3DTransform(R=R, t=t) # se3_points{i} = [R t; [0, 0, 0, 1]];
        
        if __debug__:
            e_n1_transformed = G.batch_transform_points(e_n1)
            e_n2_transformed = G.batch_transform_points(e_n2)
            
            e_n_transformed = e_n2_transformed - e_n1_transformed
            e_m_aligned = e_m2 - e_m1
            
            u_n_transformed = e_n_transformed / torch.norm(e_n_transformed, dim = -1, keepdim=True)
            u_m_aligned = e_m_aligned / torch.norm(e_m_aligned, dim = -1, keepdim=True)
            
            # Dot product of unit vectors should be 1 as they are aligned
            assert torch.allclose(torch.sum(u_n_transformed * u_m_aligned, dim=-1), torch.ones(1, device=u_n_transformed.device))
            # The starting point of e_m should be the same as the transformed e_n1
            assert torch.allclose(e_n1_transformed, e_m1)
            
        return G, G.get_matrix().transpose(-1, -2) # Transform3d: (batch_size), Tensor: (batch_size, 4, 4)
        
if __name__ == '__main__':
    main_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.abspath(os.path.join(main_dir, "../train_config.yaml"))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_loader = {}
    data_loader['train'] = torch.utils.data.DataLoader(
    dataset= Feeder(**(config["train_feeder_args"])),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    drop_last=True,)
    
    for i, data in enumerate(data_loader['train']):
        if i == 1 or i == 2 or i == 3 or i == 4:
            continue
        print("data[0].shape: ", data[0].shape) # Pose -> Tensor: [batch_size, 3, sample_cnt_pose=120, joints=21, no_of_hands=2]
        print("data[1].shape: ", data[1].shape) # RGB -> Tensor: [batch_size, sample_cnt_vid=8, rgb_feature_dim=1536]
        #print(data[2]) # label -> Tensor: [batch_size]
        #print(data[3]) # verb label -> Tensor: [batch_size]
        #print(data[4]) # noun label -> Tensor: [batch_size]
        #print(data[5]) # index -> Tensor: [batch_size]
        # Example usage
        # permute the dimensions to (batch_size, sample_cnt_pose=120, no_of_hands=2, joints=21, 3)
        hand = SE_Skeleton(data[0].permute(0, 2, 4, 3, 1).to('cuda'))
        C_t = hand.calculate_C_t() # C_t: (batch_size, sample_cnt_pose=120, no_of_pairs=(no_total_edges=no_of_hands*no_of_edges) * (no_total_edges - 1)=2070 , 4, 4)
        print("C(t).shape: ", C_t.shape)
        #break