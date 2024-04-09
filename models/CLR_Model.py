import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

# from point_4d_convolutionv1 import *
from point_4d_convolution import *
from SHOT import *
from transformer_v2 import *
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL2


def chamfer_distance(array1, array2):
    """
    Compute the Chamfer Distance between two point clouds
    Args:
    array1: (batch_size, num_points, dim) the first point cloud
    array2: (batch_size, num_points, dim) the second point cloud
    Returns:
    dist: (batch_size,) the chamfer distance between array1 and array2
    """
    batch_size, num_points, points_dim = array1.shape

    dist1 = torch.sum((array1[:, :, None, :] - array2[:, None, :, :]) ** 2, dim=-1)
    dist2 = torch.sum((array2[:, :, None, :] - array1[:, None, :, :]) ** 2, dim=-1)

    return torch.mean(torch.min(dist1, dim=1)[0], dim=1) + torch.mean(torch.min(dist2, dim=1)[0], dim=1)

class ContrastiveLearningModel(nn.Module):
    def __init__(self, radius=0.1, nsamples=32, spatial_stride=32,                            # P4DConv: spatial
                 temporal_kernel_size=3, temporal_stride=3,                                   # P4DConv: temporal
                 en_emb_dim=1024, en_depth=10, en_heads=8, en_head_dim=256, en_mlp_dim=2048,  # encoder
                 de_emb_dim=512,  de_depth=4,  de_heads=8, de_head_dim=256, de_mlp_dim=1024,  # decoder 
                 mcm_ratio = 0.7,
                 mask_ratio = 0.9,
                 num_classes=60,
                 dropout1=0.05,
                 dropout_cls=0.5,
                 pretraining=True,
                 vis=False,
                 vcm=False,
                 ):
        super(ContrastiveLearningModel, self).__init__()

        self.pretraining = pretraining
        self.mask_ratio = mask_ratio
        self.mcm_ratio = mcm_ratio

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[en_emb_dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max',
                                  pretraining=self.pretraining)

        # encoder        
        self.encoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=en_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.encoder_transformer = Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)
        self.encoder_norm = nn.LayerNorm(en_emb_dim)

        self.vis = vis
        self.vcm = vcm
        self.nsamples = nsamples
        self.tk = temporal_kernel_size
        self.encoder_temp_Transformer = temperal_Transformer(en_emb_dim, en_depth, en_heads, en_head_dim, en_mlp_dim, dropout=dropout1)

        if self.pretraining:
            
            self.shotL = getDescriptorLength(elevation_divisions=2,
                                        azimuth_divisions=4
            )
            # Mask
            self.mask_token = nn.Parameter(torch.zeros(1, 1, de_emb_dim))
            trunc_normal_(self.mask_token, std=.02)

            # decoder
            self.decoder_embed = nn.Linear(en_emb_dim, de_emb_dim, bias=True)
            self.decoder_pos_embed = nn.Conv1d(in_channels=4, out_channels=de_emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

            self.decoder_transformer = Transformer(de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1)

            self.decoder_tmp_transformer = temperal_Transformer(de_emb_dim, de_depth, de_heads, de_head_dim, de_mlp_dim, dropout=dropout1)
            self.decoder_norm = nn.LayerNorm(de_emb_dim)

            # points_predictor
            self.points_predictor = nn.Conv1d(de_emb_dim, 3 * nsamples * temporal_kernel_size, 1)
            self.shot_predictor = nn.Conv1d(de_emb_dim, self.shotL * (temporal_kernel_size-1), 1)

            # loss
            self.criterion_dist = ChamferDistanceL2().cuda()
            self.criterion_shot = torch.nn.SmoothL1Loss().cuda()



        else:
            # PointMAE mlp_head
            self.cls_token = nn.Parameter(torch.zeros(1, 1, en_emb_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, 3))

            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)



            self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(en_emb_dim),
            nn.Linear(en_emb_dim, en_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_cls),
            nn.Linear(en_mlp_dim, num_classes),
        )

            self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(en_emb_dim),
            nn.Linear(en_emb_dim, en_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_cls),
            nn.Linear(en_mlp_dim, num_classes),
        )
            
         




    def random_masking(self, x):
        B, G, _ = x.shape

        if self.mask_ratio == 0:
            return torch.zeros(x.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(x.device) # B G
    
    def dynamic_step(self, x, B, L, N, D, ori_f,pre_f):
        origin_patch_embed_vectors = ori_f.detach().clone().reshape(B, L, N, D) # [B, L, N, C]
        contra_patch_embed_vectors = pre_f.detach().clone().reshape(B, L, N, D) # [B, L, N, C]

        distance  = torch.norm(origin_patch_embed_vectors - contra_patch_embed_vectors,p=2,dim=3)# [B, L, N]
        importance = distance.flatten(1) # [B, L*N]
        
        # importance = torch.cat((distance.flatten(1), distance[:,-1,:]), dim=1) # [B, L*N]


        ids_sorted = torch.argsort(importance, dim=1, descending=True) 
        num_compressed_tokens = int((1 - self.mcm_ratio) * (N * L))
        num_input_tokens = int((1 - self.mask_ratio) * (N * L))
        noise = torch.rand(B, num_compressed_tokens, device=x.device)  # noise in [0, 1]
        noise_id_shuffled = torch.argsort(noise, dim=1)
        
        ids_sorted[:,:num_compressed_tokens] = torch.gather(ids_sorted[:,:num_compressed_tokens], dim=1, index=noise_id_shuffled)
        
        ids_restore = torch.argsort(ids_sorted, dim=1)

        
        input_mask = torch.ones([B, N * L], device=x.device)
        input_mask[:, :num_input_tokens] = 0            

        
        target_mask = torch.ones([B, N * L], device=x.device)
        if self.pretraining:
            target_mask[:, num_input_tokens:num_compressed_tokens] = 0
        else:
            target_mask[:, :num_compressed_tokens] = 0

        
        input_mask = torch.gather(input_mask, dim=1, index=ids_restore)
        target_mask = torch.gather(target_mask, dim=1, index=ids_restore)
        
        return input_mask.to(torch.bool), target_mask.to(torch.bool)

    def dynamic_masking(self, x, features, pre_features):
        '''Dynamic Masking
        Args:
            x: (B,L*N,3)
            features: (B, L, N, C)
            pre_features: (B, L, N, C)
            '''

        B, G,_ = x.shape
        _, L, N, C = features.shape
        mask ,target_mask = self.dynamic_step(x,B,L,N,C,features,pre_features)
        if self.vcm:
            masks = (mask,target_mask)
        else:
            masks = (None,target_mask)

        return masks

    def forward_encoder(self, x):
        # [B, L, N, 3]
        if self.pretraining:
            xyzs, features, xyzs_neighbors, shot_descriptors ,pre_features,pre_xyz_neighbors = self.tube_embedding(x)  
        # [B, L, N, 3] [B, L, C, N] [B, L, N, tk, nn, 3] [B, L, N, tk, shotL] [B, L, C, N] [B, L, N, tk, nn, 3]
        else:
            xyzs, features ,pre_features= self.tube_embedding(x)                                          # [B, L, n, 3], [B, L, C, n]
            # xyzs, features = self.tube_embedding2(x)                                          # [B, L, n, 3], [B, L, C, n]
        # print(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), "GB")
        
        features = features.permute(0, 1, 3, 2)                                              # [B, L, N, C]        
        batch_size, L, N, C = features.shape


        all_xyzs = torch.reshape(input=xyzs, shape=(batch_size, L*N, 3))                       # [B, L*N, 3]

        # xyzt position
        # xyzts = []
        # xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        # xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]                # L*[B, N, 3]
        # for t, xyz in enumerate(xyzs):
        #     t = torch.ones((batch_size, N, 1), dtype=torch.float32, device=x.device) * (t+1)
        #     xyzt = torch.cat(tensors=(xyz, t), dim=2)
        #     xyzts.append(xyzt)
        # xyzts = torch.stack(tensors=xyzts, dim=1)                                            # [B, L, N, 4]

        # # Token sequence
        # xyzts = torch.reshape(input=xyzts, shape=(batch_size, L*N, 4))                       # [B, L*N, 4]
        


        #获取xyzs的掩码

        if self.pretraining:
            # Targets
            xyzs_neighbors = torch.reshape(input=xyzs_neighbors, shape=(batch_size, L*N, self.tk, self.nsamples, 3)) # [B, L*N, tk, nn, 3]
            shot_descriptors = torch.reshape(input=shot_descriptors, shape=(batch_size, L*N, self.tk, self.shotL))   # [B, L*N, tk, shotL]

            # Masking
#             bool_masked_pos = self.random_masking(all_xyzs)       # [B, L*N]   Vis=0 Mask=1
#             masks = (None,bool_masked_pos)
            masks = self.dynamic_masking(all_xyzs, features, pre_features)       # [B, L*N]   Vis=0 Mask=1

            if masks[0] is None:
                bool_masked_input = masks[1]
                bool_masked_pos = masks[1]
            else:
                bool_masked_input = masks[0]
                bool_masked_pos = masks[1]
            features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            # Encoding the visible part

            fea_emb_vis = features[~bool_masked_input].reshape(batch_size, -1, C)
            # pos_emb_vis = xyzts[~bool_masked_pos].reshape(batch_size, -1, 4)

            # 可视部分的位置编码
            xyzs_emb_vis = all_xyzs[~bool_masked_input].reshape(batch_size, -1, 3) # [B, L*N-m, 3]

            
            # pos_emb_vis = self.encoder_pos_embed(pos_emb_vis.permute(0, 2, 1)).permute(0, 2, 1)

            # fea_emb_vis = fea_emb_vis + pos_emb_vis

            #特征嵌入只保留特征
            fea_emb_vis = fea_emb_vis


            fea_emb_vis = self.encoder_transformer(xyzs_emb_vis, fea_emb_vis)
            #增加时序分支
            tem_emb_vis = self.encoder_temp_Transformer(fea_emb_vis)

            fea_emb_vis = (fea_emb_vis + tem_emb_vis)/2

            fea_emb_vis = self.encoder_norm(fea_emb_vis)

            return fea_emb_vis, bool_masked_pos,bool_masked_input, xyzs_neighbors, shot_descriptors,all_xyzs

        else:
            #修改features的形状
            # print(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), "GB")
            
            if self.vcm and self.mask_ratio != 0:
                masks = self.dynamic_masking(all_xyzs, features, pre_features)
                mask = masks[1]
                features = torch.reshape(input=features, shape=(batch_size, L*N, C))
                all_xyzs = all_xyzs[~mask].reshape(batch_size, -1, 3)
                features = features[~mask].reshape(batch_size, -1, C)
            else:
                features = torch.reshape(input=features, shape=(batch_size, L*N, C))
            
                


            # xyzts = self.encoder_pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

            # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # cls_pos = self.cls_pos.expand(batch_size, -1, -1)

            # features = torch.cat((cls_tokens, features), dim=1)
            # # xyzts = torch.cat((cls_pos, xyzts), dim=1)
            # all_xyzs = torch.cat((cls_pos, all_xyzs), dim=1)
        

            # embedding = xyzts + features
            # embedding = features
            # temp_feature = features
            # print(embedding.shape)
            output = self.encoder_transformer(all_xyzs,features)
#             output = self.encoder_norm(output)
            # concat_f = torch.cat([output[:, 0], output[:, 1:].max(1)[0]], dim=-1)
            
            output2 = self.encoder_temp_Transformer(features)
            output = (output + output2)/2
            output = self.encoder_norm(output)
            output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
            # output2 = torch.max(input=output2, dim=1, keepdim=False, out=None)[0]

            output = self.mlp_head1(output)
            # output2 = self.mlp_head2(output2)
            
            # return output,output2
            return output


    # def forward_decoder(self, emb_vis, mask, xyzts, xyzs_neighbors, shot_descriptors, xyzs):
    def forward_decoder(self, emb_vis, mask, mask_input,xyzs_neighbors, shot_descriptors, xyzs):
        emb_vis = self.decoder_embed(emb_vis)
        batch_size, N_vis, C_decoder = emb_vis.shape

        # pos_emd_vis = xyzts[~mask].reshape(batch_size, -1, 4)
        # pos_emd_mask = xyzts[mask].reshape(batch_size, -1, 4)

        # pos_emd_vis = self.decoder_pos_embed(pos_emd_vis.permute(0, 2, 1)).permute(0, 2, 1)
        # pos_emd_mask = self.decoder_pos_embed(pos_emd_mask.permute(0, 2, 1)).permute(0, 2, 1)

        pos_vis = xyzs[~mask_input].reshape(batch_size, -1, 3)

        if self.vcm:
            pos_mask = xyzs[~mask].reshape(batch_size, -1, 3)
        else: 
            pos_mask = xyzs[mask_input].reshape(batch_size, -1, 3)

        # _,N_masked,_ = pos_emd_mask.shape
        _,N_masked,_ = pos_mask.shape

        # append masked tokens to sequence
        mask_tokens = self.mask_token.expand(batch_size, N_masked, -1)
        emb_all = torch.cat([emb_vis, mask_tokens], dim=1)


        # pos_all = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        pos_all = torch.cat([pos_vis, pos_mask], dim=1)

        # emb_all = emb_all + pos_all
        

        emb_all = self.decoder_transformer(pos_all, emb_all)  # [B, L*N, C]
        emb_temp_all = self.decoder_tmp_transformer(emb_all)
        emb_all = (emb_all + emb_temp_all)/2

        emb_all = self.decoder_norm(emb_all)

        masked_emb = emb_all[:, -N_masked:, :]       # [B, M, C]
        masked_emb = masked_emb.transpose(1, 2)      # [B, C, M]

        # reconstruct points
        pre_points = self.points_predictor(masked_emb).transpose(1, 2)   #[B,M,C]->[B,M,3*nsamples*tk]

        pre_points = pre_points.reshape(batch_size*N_masked, self.tk, self.nsamples, 3)                     # [B*M, tk, nn, 3]
        pred_list = torch.split(tensor=pre_points, split_size_or_sections=1, dim=1)     
        pred_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in pred_list]                     # tk*[B*M, nn, 3]

        # forward Loss
        if self.vcm:
            gt_points = xyzs_neighbors[~mask].reshape(batch_size*N_masked, self.tk, self.nsamples, 3)            # [B*M, tk, nn, 3]
        else:
            gt_points = xyzs_neighbors[mask_input].reshape(batch_size*N_masked, self.tk, self.nsamples, 3)            # [B*M', tk, nn, 3]
        gt_points_list = torch.split(tensor=gt_points, split_size_or_sections=1, dim=1)
        gt_points_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in gt_points_list]           # tk*[B*M, nn, 3]

        point_loss = 0
        for tk_i in range(self.tk):
            point_loss += chamfer_distance(pred_list[tk_i], gt_points_list[tk_i])
        point_loss = point_loss / self.tk

        # reconstruct shot
        pre_shot = self.shot_predictor(masked_emb).transpose(1, 2)   
        pre_shot = pre_shot.reshape(batch_size*N_masked, (self.tk-1), self.shotL)                           # [B*M, tk-1, shotL]
        pre_shot_list = torch.split(tensor=pre_shot, split_size_or_sections=1, dim=1)     
        pre_shot_list = [torch.squeeze(input=shot, dim=1).contiguous() for shot in pre_shot_list]           # (tk-1)*[B*M, shotL]

        if self.vcm:
            gt_shot = shot_descriptors[~mask].reshape(batch_size*N_masked, self.tk, self.shotL)                  # [B*M, tk, shotL]
        else:
            gt_shot = shot_descriptors[mask_input].reshape(batch_size*N_masked, self.tk, self.shotL)                  # [B*M, tk, shotL]
        gt_shot_list = torch.split(tensor=gt_shot, split_size_or_sections=1, dim=1)     
        gt_shot_list = [torch.squeeze(input=shot, dim=1).contiguous() for shot in gt_shot_list]             # tk*[B*M, shotL]

        shot_loss = 0
        for tk_i in range(self.tk-1):
            shot_loss += self.criterion_shot(pre_shot_list[tk_i], gt_shot_list[tk_i+1]-gt_shot_list[tk_i])
        shot_loss = shot_loss / (self.tk-1)

        # loss = point_loss + shot_loss
        loss = point_loss

        if self.vis:
            vis_points = xyzs_neighbors[~mask_input].reshape(batch_size, -1, self.tk, self.nsamples, 3) # [B, L*N-m, tk*nn, 3]
            pre_points = pre_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)           
            gt_points = gt_points.reshape(batch_size, N_masked, self.tk, self.nsamples, 3)
            return pre_points, gt_points, vis_points, mask
        else:
            return loss


    def forward(self, clips):
        # [B, L, N, 3]
        if self.pretraining:
            emb_vis, mask, mask_ran, xyzs_neighbors, shot_descriptors, xyzs = self.forward_encoder(clips)

            if self.vis:
                pre_points, gt_points, vis_points, mask = self.forward_decoder(emb_vis, mask, mask_ran, xyzs_neighbors, shot_descriptors, xyzs)
                return pre_points, gt_points, vis_points, mask
            else:
                loss = self.forward_decoder(emb_vis, mask, mask_ran,xyzs_neighbors, shot_descriptors, xyzs)
                return loss
        else:
            # output1,output2 = self.forward_encoder(clips)
            output = self.forward_encoder(clips)
            # print(output.shape)
            return output


