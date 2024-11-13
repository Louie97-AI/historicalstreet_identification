import cv2
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTFeatureExtractor
from crossattention_model import Model_CrossAttention_VIT_GCN
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, dataframe,augm=True):
        """
        Args:
            dataframe (Pandas DataFrame): DataFrame containing images and labels.

        Contents:
            Mainly includes two parts:
            1. Image Augmentation: First, image augmentation is applied, and the augmented images are placed together with the original images in gpf.
            2. Convert gdf to tensor: Split gdf into three parts:
                1. Image tensor
                2. Road index tensor
                3. Label tensor
       """
        self.dataframe = dataframe
        self.img_scale1 = None
        self.img_scale2 = None
        self.img_scale3 = None
        self.roadindex = None
        self.label = None
        
        if augm:
            self.data_augment()
            self.prepropose()
        
        else:
            self.prepropose()

    def augment_transform(self,methods,imga):
        """_summary_

        Args:
            methods (str): methods ,including "flip" and "rotate", is str type input for which judge the method to augment img.
            imga (np.ndarray): imga is the binary photo used to transform.
            return: return 3 fliped img or 3 rotated img
        """
        
        if methods == "flip":

            #horizental flip
            img1 = cv2.flip(imga,1)

            #vertical flip
            img2 = cv2.flip(imga,0)

            #horizental & vertical flip
            img3 = cv2.flip(imga, -1)
            
            return [img1,img2,img3]
        
        elif methods == "rotate":

            rotated_img1 = cv2.rotate(imga, cv2.ROTATE_90_CLOCKWISE)
            
            rotated_img2 = cv2.rotate(imga, cv2.ROTATE_180)
            
            rotated_img3 = cv2.rotate(imga, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            return [rotated_img1,rotated_img2,rotated_img3]

        elif methods == "flip_and_rotate":
            #horizental flip
            img1 = cv2.flip(imga,1)

            #vertical flip
            img2 = cv2.flip(imga,0)

            #horizental & vertical flip
            img3 = cv2.flip(imga, -1)
            
            rotated_img1 = cv2.rotate(imga, cv2.ROTATE_90_CLOCKWISE)
            
            rotated_img2 = cv2.rotate(imga, cv2.ROTATE_180)
            
            rotated_img3 = cv2.rotate(imga, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            return [img1,img2,img3,rotated_img1,rotated_img2,rotated_img3]

    def get_sigle_augment_column(self,row,scale):
        '''
        row is the gdf row
        num = -2/-3/-4
        '''
        img = row[scale][0]
        original_list = [img]

        imgs_list = self.augment_transform("flip_and_rotate",img)

        original_list.extend(imgs_list)
        
        newpd = pd.DataFrame({row.index[scale]:original_list})
        
        return newpd

    def data_augment(self):
        augment_list=[]

        for index,row in self.dataframe.iterrows():
            print("{}%".format((index/len(self.dataframe))*100))
            pd01 = self.get_sigle_augment_column(row,-2)
            pd02 = self.get_sigle_augment_column(row,-3)
            pd03 = self.get_sigle_augment_column(row,-4)

            augmented_df = pd.concat([pd03,pd02,pd01],axis=1)

            for _,original_column_id in enumerate(row.index):
                if 'Scale' not in original_column_id:
                    augmented_df[original_column_id]= row[original_column_id]

            augment_list.append(augmented_df)

        augmented_dataframe = pd.concat(augment_list,axis=0)

        new_column_order = list(self.dataframe.columns)
        df = augmented_dataframe.reindex(columns=new_column_order)
        self.dataframe = df

    def preprocess_image(self,img):

        image_array = np.stack(img*3, axis=-1)
        image = Image.fromarray(np.uint8(image_array))  
        transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(), 
        ])  

        tensor_image = transform(image)  
        tensor_image = tensor_image.unsqueeze(0)  
    
        return tensor_image  
    

    def prepropose(self):
        
        scale3_img = self.dataframe.iloc[:,-2]
        scale3_feature_tensor = torch.cat([self.preprocess_image(scale3_img.iloc[num]) for num in range(len(scale3_img))],dim=0)

        scale2_img = self.dataframe.iloc[:,-3]
        scale2_feature_tensor = torch.cat([self.preprocess_image(scale2_img.iloc[num]) for num in range(len(scale2_img))],dim=0)

        scale1_img = self.dataframe.iloc[:,-4]
        scale1_feature_tensor = torch.cat([self.preprocess_image(scale1_img.iloc[num]) for num in range(len(scale1_img))],dim=0)

        road_index = np.array(self.dataframe["ORIGIN_ROAD"])

        road_index_tensor = torch.tensor(road_index,dtype=torch.float32).unsqueeze(1)

        labels = np.array(self.dataframe["labels"])

        labels = torch.tensor(labels,dtype=torch.float32).unsqueeze(1)
        
        self.img_scale1 = scale1_feature_tensor
        self.img_scale2 = scale2_feature_tensor
        self.img_scale3 = scale3_feature_tensor
        self.roadindex = road_index_tensor
        self.label = labels
        
        return self.img_scale2


    def __len__(self):
        return len(self.dataframe)        


    def __getitem__(self, idx):
        
        img_s1 = self.img_scale1[idx]
        img_s2 = self.img_scale2[idx]
        img_s3 = self.img_scale3[idx]
        roadidx = self.roadindex[idx]
        label = self.label[idx]
        
        return {"img_s1":img_s1,
                "img_s2":img_s2,
                "img_s3":img_s3,
                "roadidx":roadidx,
                "labels":label}

