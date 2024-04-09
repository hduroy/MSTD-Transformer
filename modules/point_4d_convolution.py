import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from SHOT import *
from typing import List


class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = '+',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 pretraining: bool = False,
                 bias: bool = False):

        super().__init__()

        self.pretraining = pretraining

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]

        #another conv for pre_feature
        # conv_d_pre = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]

        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            # conv_d_pre.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
            # conv_d_pre.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)
        # self.conv_d_pre = nn.Sequential(*conv_d_pre)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]
            
        
        if self.pretraining:
            new_xyzs = []
            new_features = []
            new_xyzs_neighbors = []
            new_shot_descriptors = []

            #使用上一次循环的anchor点作为下一次循环的anchor点,保证计算相似度是是对于相邻帧的同一位置进行计算
            new_contrast_features = []
            new_contrast_xyzs_neighbors = []

            for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
                # spatial anchor point subsampling by FPS
                anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
                anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
                anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
                anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

                new_feature = []
                new_xyz_neighbor = []
                new_shot_descriptor = []

                #每次循环计算一个anchor点的特征
                new_contrast_feature = []
                new_contrast_xyz_neighbor = []

                #判断是否是第一次循环
                first = True if t == self.temporal_kernel_size//2 else False

                for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                    neighbor_xyz = xyzs[i]

                    idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)                                          # (B, N//spatial_stride, k)

                    neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                    neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                    new_xyz_neighbor.append(neighbor_xyz_grouped)

                    #如果new_xyzs不为空,则计算描述子
                    if first:
                        pre_idx = pointnet2_utils.ball_query(self.r, self.k, xyzs[i+self.temporal_kernel_size], anchor_xyz)                                # (B, N//spatial_stride, k) 
                        pre_neighbor_xyz_flipped = xyzs[i+self.temporal_kernel_size].transpose(1, 2).contiguous()                                                    # (B, 3, N)
                        pre_neighbor_xyz_grouped = pointnet2_utils.grouping_operation(pre_neighbor_xyz_flipped, pre_idx)                             # (B, 3, N//spatial_stride, k)
                        new_contrast_xyz_neighbor.append(pre_neighbor_xyz_grouped)
                    else:
                        pre_idx = pointnet2_utils.ball_query(self.r, self.k, xyzs[i-self.temporal_kernel_size], anchor_xyz)                                # (B, N//spatial_stride, k) 
                        pre_neighbor_xyz_flipped = xyzs[i-self.temporal_kernel_size].transpose(1, 2).contiguous()                                                    # (B, 3, N)
                        pre_neighbor_xyz_grouped = pointnet2_utils.grouping_operation(pre_neighbor_xyz_flipped, pre_idx)                             # (B, 3, N//spatial_stride, k)
                        new_contrast_xyz_neighbor.append(pre_neighbor_xyz_grouped)

                    # TCD
                    
                    neighs = neighbor_xyz_grouped.permute(0, 2, 3, 1)                                                                   # (B, N//spatial_stride, k, 3)
                    descriptor = getDescribe(anchor_xyz, neighs,
                                            SHOTradius=self.r,
                                            useInterpolation=True, useNormalization=False, 
                                            nnidx=idx
                    )
                    new_shot_descriptor.append(descriptor.detach())                                                                     # (B, N//spatial_stride, shotL)

                    xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                    t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                    displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                    displacement = self.conv_d(displacement)


                    #获得pre_feature
                    
                    pre_xyz_displacement = pre_neighbor_xyz_grouped - anchor_xyz_expanded    # (B, 3, N//spatial_stride, k)
                    pre_t_displacement = torch.ones((pre_xyz_displacement.size()[0], 1, pre_xyz_displacement.size()[2], pre_xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t) # (B, 1, N//spatial_stride, k)
                    pre_displacement = torch.cat(tensors=(pre_xyz_displacement, pre_t_displacement), dim=1, out=None) # (B, 4, N//spatial_stride, k)
                    # pre_displacement = self.conv_d_pre(pre_displacement)
                    pre_displacement = self.conv_d(pre_displacement)


                    if self.in_planes != 0:
                        neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                        feature = self.conv_f(neighbor_feature_grouped)

                        #获得pre_feature
                        if first:
                            pre_feature = pointnet2_utils.grouping_operation(features[i+self.temporal_kernel_size], pre_idx)                                 # (B, in_planes, N//spatial_stride, k)
                            pre_feature = self.conv_f(pre_feature)
                        else:
                            pre_feature = pointnet2_utils.grouping_operation(features[i-self.temporal_kernel_size], pre_idx)                                 # (B, in_planes, N//spatial_stride, k)
                            pre_feature = self.conv_f(pre_feature)

                        if self.operator == '+':
                            feature = feature + displacement
                            #pre_feature
                            pre_feature = pre_feature + pre_displacement
                        else:
                            feature = feature * displacement
                            #pre_feature
                            pre_feature = pre_feature * pre_displacement

                    else:
                        feature = displacement
                        #pre_feature
                        pre_feature = pre_displacement


                    feature = self.mlp(feature)

                    #如果new_features不为空,则计算pre_feature
                    
                    pre_feature = self.mlp(pre_feature)
                        
                        

                    if self.spatial_pooling == 'max':
                        feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                        pre_feature = torch.max(input=pre_feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                    elif self.spatial_pooling == 'sum':
                        feature = torch.sum(input=feature, dim=-1, keepdim=False)
                        pre_feature = torch.sum(input=pre_feature, dim=-1, keepdim=False)                
                    else:
                        feature = torch.mean(input=feature, dim=-1, keepdim=False)
                        pre_feature = torch.mean(input=pre_feature, dim=-1, keepdim=False)              

                    new_feature.append(feature)
                    #添加pre_feature
                    new_contrast_feature.append(pre_feature)

                new_feature = torch.stack(tensors=new_feature, dim=1)
                new_xyz_neighbor = torch.stack(tensors=new_xyz_neighbor, dim=-1)         # (B, 3, N//spatial_stride, k, tk)
                new_xyzs_neighbors.append(new_xyz_neighbor)

            
                new_shot_descriptor = torch.stack(tensors=new_shot_descriptor, dim=2)    # (B, N//spatial_stride, tk, shotL)
                new_shot_descriptors.append(new_shot_descriptor)

                new_contrast_feature = torch.stack(tensors=new_contrast_feature, dim=1)
                new_contrast_xyz_neighbor = torch.stack(tensors=new_contrast_xyz_neighbor, dim=-1)         # (B, 3, N//spatial_stride, k, tk)
                new_contrast_xyzs_neighbors.append(new_contrast_xyz_neighbor)

                if self.temporal_pooling == 'max':
                    new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
                    new_contrast_feature = torch.max(input=new_contrast_feature, dim=1, keepdim=False)[0]
                elif self.temporal_pooling == 'sum':
                    new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
                    new_contrast_feature = torch.sum(input=new_contrast_feature, dim=1, keepdim=False)
                else:
                    new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
                    new_contrast_feature = torch.mean(input=new_contrast_feature, dim=1, keepdim=False)
                new_xyzs.append(anchor_xyz)
                new_features.append(new_feature)

                #添加new_contrast_feature
                new_contrast_features.append(new_contrast_feature)

            new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
            new_features = torch.stack(tensors=new_features, dim=1)

            new_xyzs_neighbors = torch.stack(tensors=new_xyzs_neighbors, dim=1)      # (B, L, 3, N//spatial_stride, k, tk)
            new_xyzs_neighbors = new_xyzs_neighbors.permute(0, 1, 3, 5, 4, 2)        # (B, L, N//spatial_stride, tk, k, 3)
            
            
            new_shot_descriptors = torch.stack(tensors=new_shot_descriptors, dim=1)  # (B, L, N//spatial_stride, tk, shotL)

            new_contrast_features = torch.stack(tensors=new_contrast_features, dim=1) # (B, L, out_planes, N//spatial_stride)
            new_contrast_xyzs_neighbors = torch.stack(tensors=new_contrast_xyzs_neighbors, dim=1)      # (B, L, 3, N//spatial_stride, k, tk)
            new_contrast_xyzs_neighbors = new_contrast_xyzs_neighbors.permute(0, 1, 3, 5, 4, 2)        # (B, L, N//spatial_stride, tk, k, 3)

            
            return new_xyzs, new_features, new_xyzs_neighbors, new_shot_descriptors, new_contrast_features, new_contrast_xyzs_neighbors
        else:
            new_xyzs = []
            new_features = []
            #使用上一次循环的anchor点作为下一次循环的anchor点,保证计算相似度是是对于相邻帧的同一位置进行计算
            new_contrast_features = []


            for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
                # spatial anchor point subsampling by FPS
                anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
                anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
                anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
                anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

                new_feature = []
                #每次循环计算一个anchor点的特征
                new_contrast_feature = []


                #判断是否是第一次循环
                first = True if t == self.temporal_kernel_size//2 else False

                for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                    neighbor_xyz = xyzs[i]

                    idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)                                          # (B, N//spatial_stride, k)

                    neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                    neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)


                    #如果new_xyzs不为空,则计算描述子
                    if first:
                        pre_idx = pointnet2_utils.ball_query(self.r, self.k, xyzs[i+self.temporal_kernel_size], anchor_xyz)                                # (B, N//spatial_stride, k) 
                        pre_neighbor_xyz_flipped = xyzs[i+self.temporal_kernel_size].transpose(1, 2).contiguous()                                                    # (B, 3, N)
                        pre_neighbor_xyz_grouped = pointnet2_utils.grouping_operation(pre_neighbor_xyz_flipped, pre_idx)                             # (B, 3, N//spatial_stride, k)
                       
                    else:
                        pre_idx = pointnet2_utils.ball_query(self.r, self.k, xyzs[i-self.temporal_kernel_size], anchor_xyz)                                # (B, N//spatial_stride, k) 
                        pre_neighbor_xyz_flipped = xyzs[i-self.temporal_kernel_size].transpose(1, 2).contiguous()                                                    # (B, 3, N)
                        pre_neighbor_xyz_grouped = pointnet2_utils.grouping_operation(pre_neighbor_xyz_flipped, pre_idx)                             # (B, 3, N//spatial_stride, k)
    


                    xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                    t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                    displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                    displacement = self.conv_d(displacement)


                    #获得pre_feature
                    
                    pre_xyz_displacement = pre_neighbor_xyz_grouped - anchor_xyz_expanded    # (B, 3, N//spatial_stride, k)
                    pre_t_displacement = torch.ones((pre_xyz_displacement.size()[0], 1, pre_xyz_displacement.size()[2], pre_xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t) # (B, 1, N//spatial_stride, k)
                    pre_displacement = torch.cat(tensors=(pre_xyz_displacement, pre_t_displacement), dim=1, out=None) # (B, 4, N//spatial_stride, k)
                    # pre_displacement = self.conv_d_pre(pre_displacement)
                    pre_displacement = self.conv_d(pre_displacement)


                    if self.in_planes != 0:
                        neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                        feature = self.conv_f(neighbor_feature_grouped)

                        #获得pre_feature
                        if first:
                            pre_feature = pointnet2_utils.grouping_operation(features[i+self.temporal_kernel_size], pre_idx)                                 # (B, in_planes, N//spatial_stride, k)
                            pre_feature = self.conv_f(pre_feature)
                        else:
                            pre_feature = pointnet2_utils.grouping_operation(features[i-self.temporal_kernel_size], pre_idx)                                 # (B, in_planes, N//spatial_stride, k)
                            pre_feature = self.conv_f(pre_feature)

                        if self.operator == '+':
                            feature = feature + displacement
                            #pre_feature
                            pre_feature = pre_feature + pre_displacement
                        else:
                            feature = feature * displacement
                            #pre_feature
                            pre_feature = pre_feature * pre_displacement

                    else:
                        feature = displacement
                        #pre_feature
                        pre_feature = pre_displacement



                    feature = self.mlp(feature)

                    #如果new_features不为空,则计算pre_feature
                    
                    pre_feature = self.mlp(pre_feature)
                        
                        

                    if self.spatial_pooling == 'max':
                        feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                        pre_feature = torch.max(input=pre_feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                    elif self.spatial_pooling == 'sum':
                        feature = torch.sum(input=feature, dim=-1, keepdim=False)
                        pre_feature = torch.sum(input=pre_feature, dim=-1, keepdim=False)                
                    else:
                        feature = torch.mean(input=feature, dim=-1, keepdim=False)
                        pre_feature = torch.mean(input=pre_feature, dim=-1, keepdim=False)              

                    new_feature.append(feature)
                    #添加pre_feature
                    new_contrast_feature.append(pre_feature)

                new_feature = torch.stack(tensors=new_feature, dim=1)

                new_contrast_feature = torch.stack(tensors=new_contrast_feature, dim=1)


                if self.temporal_pooling == 'max':
                    new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
                    new_contrast_feature = torch.max(input=new_contrast_feature, dim=1, keepdim=False)[0]
                elif self.temporal_pooling == 'sum':
                    new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
                    new_contrast_feature = torch.sum(input=new_contrast_feature, dim=1, keepdim=False)
                else:
                    new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
                    new_contrast_feature = torch.mean(input=new_contrast_feature, dim=1, keepdim=False)
                new_xyzs.append(anchor_xyz)
                new_features.append(new_feature)

                #添加new_contrast_feature
                new_contrast_features.append(new_contrast_feature)

            new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
            new_features = torch.stack(tensors=new_features, dim=1)
            new_contrast_features = torch.stack(tensors=new_contrast_features, dim=1) # (B, L, out_planes, N//spatial_stride)
            
            return new_xyzs, new_features, new_contrast_features
        


class P4DTransConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 original_planes: int = 0,
                 bias: bool = False):
        """
        Args:
            in_planes: C'. when point features are not available, in_planes is 0.
            out_planes: C"
            original_planes: skip connection from original points. when original point features are not available, original_in_planes is 0.
            bias: whether to use bias
            batch_norm: whether to use batch norm
            activation:
        """
        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm

        conv = []
        for i in range(len(mlp_planes)):
            if i == 0:
                conv.append(nn.Conv1d(in_channels=in_planes+original_planes, out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            else:
                conv.append(nn.Conv1d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                conv.append(nn.BatchNorm1d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, xyzs: torch.Tensor, original_xyzs: torch.Tensor, features: torch.Tensor, original_features: torch.Tensor = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        xyzs : torch.Tensor
            (B, T, N', 3) tensor of the xyz positions of the convolved features
        original_xyzs : torch.Tensor
            (B, T, N, 3) tensor of the xyz positions of the original points
        features : torch.Tensor
            (B, T, C', N') tensor of the features to be propigated to
        original_features : torch.Tensor
            (B, T, C, N) tensor of original point features for skip connection

        Returns
        -------
        new_features : torch.Tensor
            (B, T, C", N) tensor of the features of the unknown features
        """

        T = xyzs.size(1)

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
        features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        new_xyzs = original_xyzs

        original_xyzs = torch.split(tensor=original_xyzs, split_size_or_sections=1, dim=1)
        original_xyzs = [torch.squeeze(input=original_xyz, dim=1).contiguous() for original_xyz in original_xyzs]

        if original_features is not None:
            original_features = torch.split(tensor=original_features, split_size_or_sections=1, dim=1)
            original_features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in original_features]

        new_features = []

        for t in range(T):
            dist, idx = pointnet2_utils.three_nn(original_xyzs[t], xyzs[t])

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feat = pointnet2_utils.three_interpolate(features[t], idx, weight)

            if original_features is not None:
                new_feature = torch.cat([interpolated_feat, original_features[t]], dim=1)
            new_feature = self.conv(new_feature)
            new_features.append(new_feature)

        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

if __name__ == '__main__':
    p4dconv = P4DConv(in_planes=0, mlp_planes=[16, 32, 64], mlp_batch_norm=[True, True, True], mlp_activation=[True, True, True], spatial_kernel_size=[0.1, 16], spatial_stride=4, temporal_kernel_size=5, temporal_stride=1, temporal_padding=[2, 2], temporal_padding_mode='replicate', operator='+', spatial_pooling='max', temporal_pooling='sum', bias=False)


    xyzs = torch.rand(2, 5, 1024, 3)
    new_xyzs, new_features, new_xyzs_neighbors, new_shot_descriptors, new_contrast_features, new_contrast_xyzs_neighbors = p4dconv(xyzs)
    print(new_xyzs.size(), new_features.size(), new_xyzs_neighbors.size(), new_shot_descriptors.size(), new_contrast_features.size(), new_contrast_xyzs_neighbors.size())

