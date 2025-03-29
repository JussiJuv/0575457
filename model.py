import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch_geometric.nn import ChebConv
from config import *

# Shared anatomical connections
BODY25_EDGES = [
    (0,1), (1,2), (2,3), (3,4), (2,5), (1,5), (5,6), (6,7),
    (1,8), (8,9), (8,12), (9,12), (0,15), (15,17), (0,16), (16,18),
    (9,10), (10,11), (11,24), (11,22), (22,23), (12,13), (13,14),
    (14,21), (14,19), (19,20)
]

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        kernel_size = nn.modules.utils._pair(kernel_size)
        dilation = nn.modules.utils._pair(dilation)
        self._padding = ((kernel_size[0] - 1) * dilation[0], 0)
        super().__init__(in_channels, out_channels, kernel_size, 
                        padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (0, 0, self._padding[0], 0))
        return super().forward(x)

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.align = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.conv = CausalConv2d(out_channels, 2*out_channels, (kernel_size, 1))
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x_in = self.align(x)
        x_conv = self.conv(x_in)
        return self.glu(x_conv) + x_in

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, Ks=3):
        super().__init__()
        self.temp_conv = TemporalConvLayer(in_channels, out_channels)
        self.cheb_conv = ChebConv(out_channels, out_channels, Ks)
        self.norm = nn.LayerNorm([out_channels])
        self.dropout = nn.Dropout(0.3)
        self.register_buffer('edge_index', edge_index)

    def forward(self, x):
        B, C_in, T, N = x.shape
        x_temp = self.temp_conv(x)  # [B, C_out, T, N]
        B_temp, C_out, T_temp, N_temp = x_temp.shape
        
        # Reshape with correct output channels
        x_reshaped = x_temp.permute(0, 2, 3, 1).reshape(-1, N_temp, C_out)
        x_spat = self.cheb_conv(x_reshaped, self.edge_index)
        
        # Maintain proper output dimensions
        x_out = x_spat.reshape(B, T_temp, N_temp, C_out).permute(0, 3, 1, 2)
        
        # Residual connection with channel alignment
        x_residual = x_out + x_temp  # Both [B, C_out, T, N]
        
        # Normalization and output
        x_norm = self.norm(x_residual.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(F.relu(x_norm))

class SkeletonSTGCN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=128, num_layers=3):
        super().__init__()
        self.edge_index = self.create_edges()
        
        # Explicit channel progression
        self.blocks = nn.ModuleList()
        channels = [in_channels] + [hidden_channels] * num_layers
        
        for i in range(len(channels)-1):
            self.blocks.append(STGCNBlock(channels[i], channels[i+1], self.edge_index))
        
        self.fc = nn.Linear(hidden_channels, 128)

    def create_edges(self):
        edge_index = []
        for src, dst in BODY25_EDGES:
            edge_index.extend([[src, dst], [dst, src]])
        return torch.tensor(edge_index).t().contiguous()

    def forward(self, x):
        # Input: [B, 2, T, 25]
        for block in self.blocks:
            x = block(x)  # [B, 128, T, 25] after first block
        return self.fc(x.mean(dim=[2, 3]))  # [B, 128]
    
# Fusion Components
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return (x * attn_weights).sum(dim=1)

class CrossModalFusion(nn.Module):
    def __init__(self, rgb_dim=512, skel_dim=128):
        super().__init__()
        self.query = nn.Linear(rgb_dim, skel_dim)
        self.key = nn.Linear(skel_dim, skel_dim)
        self.value = nn.Linear(skel_dim, skel_dim)
        
    def forward(self, rgb_feat, skel_feat):
        Q = self.query(rgb_feat)
        K = self.key(skel_feat)
        V = self.value(skel_feat)
        attn = torch.softmax(Q @ K.transpose(1,2), dim=-1)
        return (attn @ V).squeeze(1)

class FusionModel(nn.Module):
    def __init__(self, num_classes, num_frames=4):
        super().__init__()
        # RGB Stream
        self.rgb_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_backbone.fc = nn.Identity()
        self.temp_attention = TemporalAttention(512)
        
        # Skeleton Stream
        self.joint_emb = nn.Linear(3, 2)
        self.skeleton_net = SkeletonSTGCN()
        
        # Fusion
        self.cross_fusion = CrossModalFusion()
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_input, skeleton_input):
        # RGB Processing
        B, C, T, H, W = rgb_input.shape
        rgb_features = self.rgb_backbone(rgb_input.permute(0,2,1,3,4).flatten(0,1))
        rgb_feat = self.temp_attention(rgb_features.view(B, T, -1))
        
        # Skeleton Processing
        skeleton = skeleton_input.view(B, T, 25, 3)
        skel_emb = self.joint_emb(skeleton).permute(0, 3, 1, 2)  # [B, 2, T, 25]
        skel_feat = self.skeleton_net(skel_emb)
        
        # Cross-modal Fusion
        fused_skel = self.cross_fusion(rgb_feat.unsqueeze(1), skel_feat.unsqueeze(1))
        
        # Final Prediction
        combined = torch.cat([rgb_feat, fused_skel], dim=1)
        return self.fc(combined)