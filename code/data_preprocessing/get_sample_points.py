import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import LineString,Point
import matplotlib.pyplot as plt
from pyproj import CRS
from tqdm import tqdm

def get_sample_points(interval_len,roads):
 
    sampling_interval = interval_len

    sample_points = []  

    for index, row in roads.iterrows():
        
        percentage = (index / 3697) * 100 
        print(f"{percentage:.2f}%")
        
        line = row['geometry']  

        line_length = line.length  

        num_points = int(line_length / sampling_interval) + 1  
        if num_points > 1:

            for i in range(num_points):  
 
                fraction = i / (num_points - 1)  

                point = line.interpolate(fraction, normalized=True)  
                sample_points.append(Point(point.x, point.y))  

    sample_points_gdf = gpd.GeoDataFrame({'geometry': sample_points})
    
    sample_points_gdf.reset_index(inplace=True)
    
    sample_points_gdf = sample_points_gdf.drop_duplicates(subset="geometry")
    
    return sample_points_gdf


