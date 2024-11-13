import os
os.environ['USE_PYGEOS'] = '0'
import cv2
import yaml
import pickle
import argparse
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pyproj import CRS
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import LineString,Point,Polygon
#
from tiff_clip import clip
from get_sample_points import get_sample_points
from graph_structure import paird_road_point,paird_road_road
from centrality_calculation import centrality2dataframe,pair_road,centrality_calculation,mergedroad2graph

parser = argparse.ArgumentParser(description='An example')

parser.add_argument(
    "--dataconfig",
    default="recover_yuan_road\configs\datadealing_config.yaml")

args = parser.parse_args()

def plot(dataframe,index,column):
    image = Image.fromarray(dataframe.iloc[index][column][0])
    image.show()
    

#photos
# Use this overall test dataset to sample images across the entire city directly following the road network
def main_test(args,motai_3=False):
    
    config_path = args.dataconfig
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    roads = gpd.read_file(config['path']["road_path"])
    roads.reset_index(inplace=True)

    ICN_roads = gpd.read_file(config['path']['merged_road_path'])
    
    samplepoints = get_sample_points(config["args"]['samplepoints_interval_len'],ICN_roads)

    img_src = rasterio.open(config['path']['tiff_path'])

    for index,item in enumerate(config['args']['scale']):
        samplepoints,count_255,count_0 = clip(samplepoints,img_src,item,contour = False)
    
    if motai_3:
        contour_src = rasterio.open(config['path']['contour_path'])
        samplepoints,count_255,count_0= clip(samplepoints,contour_src,224,contour = True)

    multi_scale_samplepoints = samplepoints
    multi_scale_samplepoints.drop(columns="index",inplace=True)
    multi_scale_samplepoints.reset_index(inplace=True)

    point_file = multi_scale_samplepoints[["index","geometry"]]

    paired_road_point_gdf = paird_road_point(roads,point_file,"points")
    paired_r_p_gdf = pd.merge(paired_road_point_gdf,multi_scale_samplepoints,left_on="CONNECTED_POINTS",right_on="index",how="left")
    
    if motai_3:
        with open(r'recover_yuan_road\official_data\1937map\research_using_data\beforedataset_data_3_Motai\paired_r_p_gdf_test_interval50_100_224_512_contour','wb') as f:
            pickle.dump(paired_r_p_gdf,f)
    else:
        with open(r'recover_yuan_road\official_data\1937map\research_using_data\beforedataset_data_2_Motai\paired_r_p_gdf_test_interval50_100_224_512','wb') as f:
            pickle.dump(paired_r_p_gdf,f)
        
    #Construct a Road Graph
    paired_r_r_gdf = paird_road_road(roads)
    with open(r'recover_yuan_road\official_data\1937map\research_using_data\beforedataset_data_3_Motai\paired_r_r_gdf_train_interval50_100_224_512_contour','wb') as f:
        pickle.dump(paired_r_r_gdf,f)
    
    return paired_r_p_gdf,paired_r_r_gdf




# Use this overall training dataset to sample images across the entire city directly along the road network.
def main_train(args,samplepoint_for_train,motai_3=False):
    
    config_path = args.dataconfig
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    roads = gpd.read_file(config['path']["road_path"])
    roads.reset_index(inplace=True)

    img_src = rasterio.open(config['path']['tiff_path'])

    for index,item in enumerate(config['args']['scale']):
        samplepoints,count_255,count_0 = clip(samplepoint_for_train,img_src,item,contour = False)
    
    if motai_3:

        contour_src = rasterio.open(config['path']['contour_path'])
        samplepoints,count_255,count_0= clip(samplepoints,contour_src,224,contour = True)

    multi_scale_samplepoints = samplepoints
    # multi_scale_samplepoints.drop(columns="index",inplace=True)
    multi_scale_samplepoints.reset_index(inplace=True)

    point_file = multi_scale_samplepoints[["index","geometry"]]

    paired_road_point_gdf = paird_road_point(roads,point_file,"points")
    paired_r_p_gdf = pd.merge(paired_road_point_gdf,multi_scale_samplepoints,left_on="CONNECTED_POINTS",right_on="index",how="left")
    
    columns_order = ['ORIGIN_ROAD', 'CONNECTED_POINTS', 'geometry_x', 'index',
       'geometry_y', 'pixel_X', 'pixel_Y', 'Scale_100', 'Scale_224',
       'Scale_512','labels']
    
    paired_r_p_gdf = paired_r_p_gdf.reindex(columns=columns_order)
    

    
    with open(r'recover_yuan_road\official_data\1937map\research_using_data\0419new_data_2Motai\0419paired_r_p_gdf_train&val_interval_50_scale_100_224_512.pickle','wb') as f:
        pickle.dump(paired_r_p_gdf,f)
    
    return paired_r_p_gdf

# get_road_feature
def get_node_feature(args):
    
    config_path = args.dataconfig
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #
    road = gpd.read_file(config['path']['road_path'])
    road=gpd.GeoDataFrame(road['geometry']).reset_index()
    #
    merged_road = gpd.read_file(config['path']['merged_road_path'])
    merged_road.drop(columns=['FID'],inplace=True)
    merged_road.reset_index(inplace=True)
    print(merged_road.shape)

    merged_road_graphed = mergedroad2graph(merged_road,config)
    print(merged_road_graphed.shape)

    multicentrality_df = centrality_calculation(merged_road_graphed,merged_road,config)
    
    road_features = pair_road(road,multicentrality_df)
    
    with open(config['path']['road_feature_path'],'wb') as f:
        pickle.dump(road_features,f)
        
    road_features.to_file(config['path']['road_feature_shp_path'], encoding="utf-8")
    
    return road_features

if __name__ == "__main__":
    
    samplepoint_for_train = gpd.read_file(r'recover_yuan_road\official_data\1937map\37map_samplepoints\true_false_points0418\new_points_dataset0419.shp')
    samplepoint_for_train.drop(columns='FID_1',inplace=True)
    paired_r_p_gdf = main_train(args,samplepoint_for_train,motai_3=False)
