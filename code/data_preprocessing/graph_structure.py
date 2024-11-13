import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import LineString, Point, Polygon
from get_sample_points import get_sample_points
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor,ViTFeatureExtractor


def paird_road_point(roads,gdf2,gdf2_property):

    gdf2_buffer = gdf2.copy()
    gdf2_buffer['geometry'] = gdf2_buffer['geometry'].buffer(5)
    
    paired_road = []
    for index,row in gdf2_buffer.iterrows():
        print(f'road and points paird finished {(index/len(gdf2_buffer))*100}%.')
        intersects_road_gdf = roads[row.geometry.intersects(roads.geometry)]
        if len(intersects_road_gdf)>0:
            paired_index = list(intersects_road_gdf["index"])
            paired_road.append([row['index'],paired_index[0]])

    paired_road_array = np.array(paired_road)

    paired_road_gdf = gpd.GeoDataFrame({"CONNECTED_POINTS":paired_road_array[:,0],
                                        "ORIGIN_ROAD":paired_road_array[:,1]})
                                        
    
    merged_gdf = pd.merge(paired_road_gdf,gdf2, left_on='CONNECTED_POINTS', right_on='index', how='left')
    gpd01 = gpd.GeoDataFrame(merged_gdf[["ORIGIN_ROAD","CONNECTED_POINTS","geometry"]])
    gpd01.set_geometry("geometry")

    return gpd01

def paird_road_road(roads):
    paired_road = []
    for index,row in roads.iterrows():
        intersect_series = roads[row.geometry.intersects(roads.geometry)]["index"]
        intersect_series = list(intersect_series)
        if len(intersect_series)==1 and intersect_series[0] != index:
            paired_road.append([row["index"],intersect_series[0]]) 
            
        elif len(intersect_series)>1:
            paired_list = [[row["index"],intersect_series[i]] for i in range(len(intersect_series)) if intersect_series[i] != row["index"]]
            paired_road.extend(paired_list)

    road_road_pair = np.array(paired_road)
    
    return road_road_pair

def plot(*args):
    type_judge = isinstance(imga,np.ndarray)
    if type_judge:
        cv2.imshow("Image",args[0])
    else:
        if len(args)>1:
            fig, ax = plt.subplots()
            args[0].plot(ax=ax)
            args[1].plot(ax=ax)
            ax.set_aspect('equal')
            
        else:
            args[0].plot()
            plt.show()
















































































#存在的三个问题：
# 1.采样得到的point会有重复情况，直接用drop_duplicates去除，已经检查过的，没有任何信息损失，去重效果很好。
# 2.采样得到的点，与道路不是完全重合的，在用intersects求几何相交时候，就会有问题，因此要用点的buffer来求相交，然后返回点的id和道路id。
# 3.十字路口的点随机给分配一个相交道路的特征。

#明天任务：
# 1.将道路对偶数据处理出来。2.给道路节点随机初始化特征。
# 2.制作好dataset，看看dataset的原理到底是咋回事，为啥bb可以将数据处理都放进去？
# 3.跑通模型。
# 4.将数据预处理部分的代码再整理一下，改正 get points模块的错误，最好将三个数据处理file检查后放在一个文件里，
# 未来只要掉这个文件，就能出来  1.包含道路和节点空间匹配信息的图片数据；2.图结构数据。






# #1.将单通道图像扩展为3通道。 
# triplet_image = np.stack([img]*3, axis=-1)

# # 2.将NumPy数组转换为PIL图像  
# triplet_image_pil = Image.fromarray(triplet_image)  
# triplet_image_pil.show()

# # 3.压缩图像信息
# # 创建一个变换，将图像从1024x1024缩放到224x224  
# resize_transform = transforms.Resize((224, 224)) 
# zip_img = resize_transform(triplet_image_pil)
# zip_img.show()
# # np01 = np.array(zip_img)

# # 4.创建tensor
# # 将 PIL 图像或 NumPy ndarray 转换为 FloatTensor。
# # 将图像的像素值范围从 [0, 255] 缩放到 [0.0, 1.0]。
# # 将图像数据的维度从 (H x W x C)（高度 x 宽度 x 通道数）转换为 (C x H x W)。
# tensor_transform = transforms.ToTensor()
# img_tensor = tensor_transform(zip_img)
# img_tensor = img_tensor.unsqueeze(0)

# # features = model(img_tensor)
# # features[0][:,0,:]

# #method02
# #VIT图像处理
# #加载VIT图像处理器，VIT处理器只能接收PIL文件，因此
# from transformers import ViTModel, ViTImageProcessor,ViTFeatureExtractor
# model_dict = r"recover_yuan_road\model\models\VITmodelweight"
# model = ViTModel.from_pretrained(model_dict)
# feature_extractor = ViTImageProcessor.from_pretrained(model_dict)
# feature_extractor2 = ViTFeatureExtractor.from_pretrained(model_dict)

# result = feature_extractor(zip_img)
# result2 = feature_extractor2(zip_img)

# result["pixel_values"][0]
# result2["pixel_values"][0]


    
    #---------------------------------------------------------------------------------------
    # #对于gdf检查指定列的重复项
    # duplicates = points.duplicated(subset=['geometry'])
    # #取出重复的点要素
    # duplicated_p = points[duplicates]
    # #对于全部的点要素进行去重 points_remove_duplicate
    # points = points.drop_duplicates(subset=['geometry'])
    

    # 得到一个road和road相互pair的gdf

    
    
    

    
    

    





