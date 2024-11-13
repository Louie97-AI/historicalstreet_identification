import yaml
import timm
import math
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import tensor
import torch.optim as optim
import torch.nn.init as init
from transformers import AutoModel
import torch.nn.functional as nnf
import torchvision.models as models
from torch_geometric.data import Data
from typing import Optional, Tuple, Type
from torch_geometric.nn import GCNConv, GATConv
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTFeatureExtractor

class CrossAttentionMerging_Block(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            depth: int = 3,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True) -> None:

        super(CrossAttentionMerging_Block, self).__init__()
        '''
        imput_data: (batch size,features)
        '''
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer)

            self.blocks.append(block)
            
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def positional_encoding(self, seq_len, d_model):

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        for blk in self.blocks:
            # x = blk(x)
            x = blk(x, y)

        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False) -> None:

        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos)

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, y)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias: bool = True,
            use_rel_pos: bool = False) -> None:

        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert (self.head_dim * self.num_heads == dim)

        # qkv = (768,768); proj = (768,768)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        #
        queries = x
        keys = y
        values = y

        #
        QB = queries.shape[0]
        KB = keys.shape[0]
        VB = values.shape[0]

        #
        Q = self.Wq(queries)
        K = self.Wk(keys)
        V = self.Wv(values)

        queries = Q.reshape(QB, self.num_heads, self.head_dim)
        keys = K.reshape(KB, self.num_heads, self.head_dim)
        values = V.reshape(VB, self.num_heads, self.head_dim)

        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        attention_logits = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = nnf.softmax(attention_logits * self.scale, dim=-1)
        attention_output = torch.matmul(attention_weights, values).permute(1, 0, 2)

        out = attention_output.reshape(QB, -1)
        out = self.proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super(MLPBlock, self).__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()

        # o_road_dim, road_dim = config['model']['GCN']['o_road_dim'],config['model']['GCN']['road_dim']

        self.linear = nn.Linear(4, 128)
        self.ln1 = nn.LayerNorm(128)

        self.conv1 = GATConv(128, 32, heads=8)
        self.ln2 = nn.LayerNorm(256)

        self.conv2 = GATConv(256, 256, heads=1)

        self.gelu = nn.GELU()

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):

        init.xavier_uniform_(self.linear.weight)
        init.constant_(self.linear.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.linear(x)
        x = self.ln1(x)
        x = self.gelu(x)

        x = self.conv1(x, edge_index)
        x = self.ln2(x)
        x = self.gelu(x)

        x = self.conv2(x, edge_index)
        # x = self.bn3(x)
        # x = self.gelu(x)  

        return x


# Perform cross-attention on the two scales separately, then concatenate them.
# class Motai_merge01(nn.Module):
#     def __init__(self,config):
#         super(Motai_merge01,self).__init__()

#         #motai1 scale2 projection 768->256
#         self.linear_proj0 = nn.Linear(768,256)

#         # merge scale2 + spatial_img
#         self.cross_attention_merge_block1 = CrossAttentionMerging_Block(embed_dim=768)
#         self.linear_proj1 = nn.Linear(768,128)

#         # merge scale1 + spatial_img
#         self.cross_attention_merge_block2 = CrossAttentionMerging_Block(embed_dim=768)
#         self.linear_proj2 = nn.Linear(768,128)

#         self.gelu = nn.GELU()
#         # self.dropout = nn.Dropout(p=0.5)

#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self,*args):

#         f1,f2,spatial_img=args[0],args[1],args[2]
#         cut = self.linear_proj0(f2)

#         #(10,768)
#         f1 = self.cross_attention_merge_block1(f1,spatial_img)
#         f1 = self.linear_proj1(f1)

#         #(10,768)
#         f2 = self.cross_attention_merge_block1(f2,spatial_img)
#         f2 = self.linear_proj1(f2)

#         merged_feature = torch.cat((f1,f2,cut),dim=1)

#         return merged_feature


# Multi-scale early fusion (Ablation experiment version).
# class Motai_merge01(nn.Module):
#     def __init__(self,config):
#         super(Motai_merge01,self).__init__()

#         # merge scale2 + spatial_img
#         self.cross_attention_merge_block1 = CrossAttentionMerging_Block(embed_dim=768)

#         # merge scale1 + spatial_img
#         self.cross_attention_merge_block2 = CrossAttentionMerging_Block(embed_dim=768)

#         # proj
#         self.linear_proj = nn.Linear(768*2,768)
#         self.gelu = nn.GELU()
#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self,*args):

#         f1,f2,spatial_img=args[0],args[1],args[2]

#         #(10,768)
#         f1 = self.cross_attention_merge_block1(f1,spatial_img)

#         #(10,768)
#         f2 = self.cross_attention_merge_block1(f2,spatial_img)

#         merged_feature = torch.cat((f1,f2),dim=1)
#         merged_feature = self.linear_proj(merged_feature)

#         return merged_feature

# Multi-scale intrinsic fusion.
class Motai_merge01(nn.Module):
    def __init__(self, config):
        super(Motai_merge01, self).__init__()

        # motai1 scale2 projection 768->256
        self.linear_proj0 = nn.Linear(768, 384)
        # self.linear_proj1 = nn.Linear(768,384)

        # merge scale1+scale2 feature
        self.cross_attention_merge_block = CrossAttentionMerging_Block(embed_dim=768)

        # # merge scale1 + spatial_img
        # self.cross_attention_merge_block2 = CrossAttentionMerging_Block(embed_dim=768)
        # self.linear_proj2 = nn.Linear(768,128)

        self.gelu = nn.GELU()
        # self.dropout = nn.Dropout(p=0.5)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, *args):

        f1, f2, spatial_img = args[0], args[1], args[2]
        # (32,384)
        f1 = self.gelu(self.linear_proj0(f1))
        cut = self.linear_proj0(f2)
        ff2 = self.gelu(cut)
        # (32,768)
        merged_feature = torch.cat((f1, ff2), dim=1)

        # (10,768)
        cross_scale_features = self.cross_attention_merge_block(merged_feature, spatial_img)

        return cross_scale_features


# Early fusion of multiple modalities (Ablation experiment version).
# class Motai_merge02(nn.Module):
#     def __init__(self, config):
#         super(Motai_merge02, self).__init__()

#         # 有参数可学习部分
#         # self.merged02_linear1 = nn.Linear(512,512)
#         # self.merged02_linear1 = nn.Linear(768,512)
#         # self.merged02_linear2 = nn.Linear(256,512)

#         # self.attention_merge_block = CrossAttentionMerging_Block(embed_dim=1024)
#         self.adaptive_module0 = nn.Linear(768, 256)
#         self.adaptive_module1 = nn.Linear(256, 256)
#         self.adaptive_module2 = nn.Linear(512, 256)
#         # self.adaptive_module2 = nn.Linear(1024,512)

#         # 没有参数可学习的部分
#         self.gelu = nn.GELU()
#         # self.dropout = nn.Dropout(p=0.5)

#         # Xavier均匀分布初始化
#         self.initialize_weights()

#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, *args):

#         f1, f2 = args[0], args[1]
        
#         f1 = self.gelu(self.adaptive_module0(f1))
#         f2 = self.gelu(self.adaptive_module1(f2))
        
#         merged_f = torch.cat((f1, f2), dim=1)

#         return merged_f


# Multi-modal extrinsic fusion.
class Motai_merge02(nn.Module):
    def __init__(self,config):
        super(Motai_merge02,self).__init__()

        # self.merged02_linear1 = nn.Linear(512,512)
        self.merged02_linear1 = nn.Linear(768,512)
        self.merged02_linear2 = nn.Linear(256,512)

        # self.attention_merge_block = CrossAttentionMerging_Block(embed_dim=1024)

        self.adaptive_module1 = nn.Linear(1024,512)
        # self.adaptive_module2 = nn.Linear(1024,512)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,*args):

        f1,f2 = args[0],args[1]
        a = self.gelu(self.merged02_linear1(f1))
        b = self.gelu(self.merged02_linear2(f2))

        #concat 1024
        merged_f = torch.cat((a,b),dim=1)

        result = self.adaptive_module1(merged_f)

        return result


class FFN(nn.Module):
    def __init__(self, merged02_output_dim):
        super(FFN, self).__init__()

        self.ffn_sequential = nn.Sequential(nn.Linear(merged02_output_dim, 256),
                                            nn.GELU(),
                                            nn.Linear(256, 2))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, data):
        output = self.ffn_sequential(data)
        return output


# --------------------------------------main model----------------------------------------------------------
# cross attention model
class Model_CrossAttention_VIT_GCN(nn.Module):
    def __init__(self, config):
        super(Model_CrossAttention_VIT_GCN, self).__init__()

        vit_path = config['choosing_model']['model']['vit']

        self.pretrain_model = ViTModel.from_pretrained(vit_path)
        # self.pretrain_resnet = Resnet101Pretrained(config)
        self.road_GAT = GAT(config)

        self.merge01 = Motai_merge01(config)

        self.merge02 = Motai_merge02(config)

        self.final_classify_head = FFN(768)

    def forward(self, img1, img2, spatial_img, index, data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]
        f2 = self.pretrain_model(img2)[0][:, 0, :]

        f4 = self.merge01(f1, f2, spatial_img)

        road = self.road_GAT(data)  # (roads,featurs)

        torch_f4_index = torch.cat((f4, index.float()), dim=1)
        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        final_features = self.merge02(f4, road_f)

        output = self.final_classify_head(final_features)

        return output


# ----------------------------------comparing different merging methods--------------------------------------

# Addition
class Addition(nn.Module):
    def __init__(self, config):
        super(Addition, self).__init__()

        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.road_GCN = GAT(config)
        # self.merge01 = Motai_merge01(config)
        # self.merge02=Motai_merge02(config)

        self.proj = nn.Linear(768, 256)
        self.ffn = nn.Linear(256, 2)

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]
        f2 = self.pretrain_model(img2)[0][:, 0, :]
        f4 = f1 + f2

        f4 = self.proj(f4)
        road = self.road_GCN(data)  # (roads,featurs)

        torch_f4_index = torch.cat((f4, index.float()), dim=1)
        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        f6 = f4 + road_f

        output = self.ffn(f6)

        return output


# concate
class Concat(nn.Module):
    def __init__(self, config):
        super(Concat, self).__init__()

        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.road_GCN = GAT(config)
        # self.merge01 = Motai_merge01(config)
        # self.merge02=Motai_merge02(config)

        self.ffn = nn.Sequential(nn.Linear(768 * 2 + 256, 1000),
                                 nn.GELU(),
                                 nn.Linear(1000, 256),
                                 nn.GELU(),
                                 nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]
        f2 = self.pretrain_model(img2)[0][:, 0, :]
        f4 = torch.concat((f1, f2), dim=1)

        road = self.road_GCN(data)  # (roads,featurs)

        torch_f4_index = torch.cat((f4, index.float()), dim=1)

        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        f6 = torch.concat((f4, road_f), dim=1)

        output = self.ffn(f6)

        return output


# ------------------------------------ abliation experiment-----------------------------------


# abliation
# scale1+GCN
class Scale1_GCN(nn.Module):
    def __init__(self, config):
        super(Scale1_GCN, self).__init__()

        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.road_GCN = GAT(config)
        # self.merge01 = Motai_merge01(config)
        self.merge02 = Motai_merge02(config)

        # self.img_projection = nn.Linear(768,512)

        self.motai2_classify_head = nn.Sequential(nn.Linear(512, 256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]
        # f1 = self.img_projection(f1)

        road = self.road_GCN(data)  # (roads,featurs)

        torch_f4_index = torch.cat((f1, index.float()), dim=1)
        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        f6 = self.merge02(f1, road_f)
        output = self.motai2_classify_head(f6)

        return output


# scale2+GCN
class Scale2_GCN(nn.Module):
    def __init__(self, config):
        super(Scale2_GCN, self).__init__()
        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)
        self.road_GCN = GAT(config)
        # self.merge01 = Motai_merge01(config)
        self.merge02 = Motai_merge02(config)

        # self.img_projection = nn.Linear(768,512)

        self.motai2_classify_head = nn.Sequential(nn.Linear(512, 256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        # f1 = self.VIT_block(img1)[0][:,0,:]
        f2 = self.pretrain_model(img2)[0][:, 0, :]
        # f2 = self.img_projection(f2)

        road = self.road_GCN(data)  # (roads,featurs)

        torch_f4_index = torch.cat((f2, index.float()), dim=1)
        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        f6 = self.merge02(f2, road_f)

        output = self.motai2_classify_head(f6)

        return output


# img1 motai
class img1_only_model(nn.Module):
    def __init__(self, config):
        super(img1_only_model, self).__init__()
        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.classfiy_head = nn.Sequential(nn.Linear(768, 256),
                                           nn.GELU(),
                                           nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]

        output = self.classfiy_head(f1)

        return output


# img2 motai
class img2_only_model(nn.Module):
    def __init__(self, config):
        super(img2_only_model, self).__init__()
        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.classfiy_head = nn.Sequential(nn.Linear(768, 256),
                                           nn.GELU(),
                                           nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):
        f2 = self.pretrain_model(img2)[0][:, 0, :]

        output = self.classfiy_head(f2)

        return output


# img1 img2 motai
class img1_img2_model(nn.Module):
    def __init__(self, config):
        super(img1_img2_model, self).__init__()
        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.merge01 = Motai_merge01(config)

        self.classfiy_head = nn.Sequential(nn.Linear(768, 256),
                                           nn.GELU(),
                                           nn.Linear(256, 2))

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, spatial_img, index, graph_data):
        f1 = self.pretrain_model(img1)[0][:, 0, :]
        f2 = self.pretrain_model(img2)[0][:, 0, :]

        f4 = self.merge01(f1, f2, spatial_img)

        output = self.classfiy_head(f4)

        return output


# road structure motai
class Road_Structure_only_model(nn.Module):
    def __init__(self, config):
        super(Road_Structure_only_model, self).__init__()

        vit_path = config['choosing_model']['model']['vit']
        self.pretrain_model = ViTModel.from_pretrained(vit_path)

        self.road_GCN = GATConv(config)
        # self.merge01 = Motai_merge01(config)
        # self.merge02=Motai_merge02(config)

        self.classfiy_head = nn.Linear(256, 2)

    # def forward(self,img2,index,data):
    def forward(self, img1, img2, index, data):

        road = self.road_GCN(data)  # (roads,featurs)

        index1 = index.long().squeeze(1)
        selected_rows = road[index1]
        road_f = selected_rows

        output = self.classfiy_head(road_f)

        return output








