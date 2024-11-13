import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import LineString,Point,Polygon
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

def distance(coord1,coord2):
    x1,y1 = coord1[0]
    x4,y4 = coord1[1]
    x2,y2 = coord2[0]
    x3,y3 = coord2[1]

    d1 = np.sqrt((x1-x2)**2+(y1-y2)**2)
    d2 = np.sqrt((x1-x3)**2+(y1-y3)**2)
    d3 = np.sqrt((x4-x2)**2+(y4-y2)**2)
    d4 = np.sqrt((x4-x3)**2+(y4-y3)**2)
    distance = [d1,d2,d3,d4]
    if min(distance) == d1:
        return (x4,y4),(x1,y1),(x3,y3)
    elif min(distance) == d2:
        return (x4,y4),(x1,y1),(x2,y2)
    elif min(distance) == d3:
        return (x1,y1),(x4,y4),(x3,y3)
    elif min(distance) == d4:
        return (x1,y1),(x4,y4),(x2,y2)

def angle(p1,p2,p3):
    vector1 = (p1[0]-p2[0],p1[1]-p2[1])
    vector2 = (p3[0]-p2[0],p3[1]-p2[1])
    vector12 = np.dot(vector1,vector2)
    dv1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
    dv2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
    cos_theta = vector12/(dv1*dv2)
    
    # 计算角度（弧度制）
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 将弧度转换为度
    angle_deg = np.degrees(angle_rad)

    return angle_deg

#angle judge. 
def angle_judge(gpd_all,idx,idxx):
    idx_x = gpd_all.loc[idx].geometry.xy[0]
    idx_y = gpd_all.loc[idx].geometry.xy[1]
    idx_coords = [(idx_x[0],idx_y[0]),(idx_x[1],idx_y[1])]
    
    idxx_x = gpd_all.loc[idxx].geometry.xy[0]
    idxx_y = gpd_all.loc[idxx].geometry.xy[1]
    idxx_coords = [(idxx_x[0],idxx_y[0]),(idxx_x[1],idxx_y[1])]
    
    p1,p2,p3 = distance(idx_coords,idxx_coords)
    angle_judge = angle(p1,p2,p3)
    return angle_judge

def icn_line(idx,gpd_all):
    intersect_idx_final=[]
    intersect_list=[]
    intersect_lines = gpd_all[gpd_all.loc[idx].geometry.intersects(gpd_all.geometry)]
    if len(intersect_lines)>0:
        intersect_idx = list(intersect_lines.index)
        for idxx in intersect_idx:
            anglee = angle_judge(gpd_all,idx,idxx)
            if anglee > 150:
                intersect_idx_final.append(idxx)
        if idx in intersect_idx_final:
            intersect_idx_final.remove(idx)
        
        return intersect_idx_final

def find_icn_lines(idx,gpd_all):
    line_list = [idx]
    intersect_list=[idx]
    while len(intersect_list)>0:
        sigle_idx = intersect_list[0]
        icn_line_list = icn_line(sigle_idx,gpd_all)
        line_list.extend(icn_line_list)
        intersect_list.extend(icn_line_list)
        intersect_list.remove(sigle_idx)
        intersect_list = list(OrderedDict.fromkeys(intersect_list))
        gpd_all.drop(index = sigle_idx,inplace=True)
            
    return line_list

def main(gpd1):
    gpd_all = gpd1.copy()
    gpd2 = gpd1.copy()
    merged_list_line_index = []
    merged_lines = []
    while len(gpd2.index)>0:
        if len(gpd2)>0:
            idx = gpd2.index[0]
            l1 = find_icn_lines(idx,gpd_all)
            l1 = list(set(l1))
            gpd2 = gpd2.drop(l1)
            merged_list_line_index.append(l1)
    
    for i in merged_list_line_index:
        merged_lines.append(gpd1.loc[i].geometry.unary_union)
    
    merged_gpd = gpd.GeoDataFrame({"geometry":merged_lines})
    
    return merged_gpd
    
if __name__ == "__main__":
    
    #data
    # gpd1 = gpd.read_file(r"recover_yuan_road\official_data\1937map\37map_road_shp\road_splited_at_points.shp")
    gpd1 = gpd.read_file(r"heat_map\remote-file\attention_map_shp\split_At_line0410.shp")

    merged_lines = main(gpd1)
    # 将 GeoDataFrame 保存为 Shapefile 文件
    output_shapefile_path = r'heat_map\remote-file\attention_map_shp\connected_clip0410.shp'

    merged_lines.to_file(output_shapefile_path, driver='ESRI Shapefile')