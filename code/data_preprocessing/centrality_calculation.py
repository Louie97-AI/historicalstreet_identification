import yaml
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import LineString,Point,polygons

#
parser = argparse.ArgumentParser()
parser.add_argument('--dataconfig',default='recover_yuan_road\configs\datadealing_config.yaml')

args = parser.parse_args()
#
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

#
def centrality2dataframe(dict,name):

    data_list = list(dict.items())
    df = pd.DataFrame(data_list, columns=['mergedroad_index', name+'_centrality'])
    return df.sort_values(by='mergedroad_index')

#
def pair_road(road,multicentrality_df):
    road_buffer = road.copy()
    road_buffer['geometry'] = road_buffer['geometry'].buffer(1)    
    
    recostrusted_road_df = pd.DataFrame()

    for index,row in road_buffer.iterrows():
        print(f'running {index/len(road_buffer)*100}%.')
        
        recostrusted_road_df.loc[index,'index'] = int(index)
        intersect_df = multicentrality_df[row['geometry'].intersects(multicentrality_df.geometry)]
        if len(intersect_df)>1:
            recostrusted_road_df['degree_centrality'] = intersect_df['degree_centrality'].mean()
            recostrusted_road_df['closeness_centrality'] = intersect_df['closeness_centrality'].mean()
            recostrusted_road_df['betweenness_centrality'] = intersect_df['betweenness_centrality'].mean()
            recostrusted_road_df['eigenvector_centrality'] = intersect_df['degree_centrality'].mean()

        elif len(intersect_df) == 1:
            recostrusted_road_df['degree_centrality'] = intersect_df['degree_centrality']
            recostrusted_road_df['closeness_centrality'] = intersect_df['closeness_centrality']
            recostrusted_road_df['betweenness_centrality'] = intersect_df['betweenness_centrality']
            recostrusted_road_df['eigenvector_centrality'] = intersect_df['degree_centrality']

    order = ['index','degree_centrality','closeness_centrality','betweenness_centrality','eigenvector_centrality']
    recostrusted_road_df = recostrusted_road_df[order]
    recostrusted_road_df['index'] = recostrusted_road_df['index'].astype(int)
    new_centrality_df = pd.merge(recostrusted_road_df,road,left_on='index',right_on='index')

    return new_centrality_df
    
def centrality_calculation(merged_road_graphed,merged_road,config):
    print('centrality_calculation_start')
    G = nx.Graph()
    G.add_edges_from(merged_road_graphed)
    nx.write_gexf(G, config['path']['road_feature_gephi_path1'])
    degree_centrality = nx.degree_centrality(G)
    d_df = centrality2dataframe(degree_centrality,'degree')
    print('d_df,done')
    
    # 计算介数中心性（Betweenness Centrality）
    betweenness_centrality = nx.betweenness_centrality(G)
    b_df = centrality2dataframe(betweenness_centrality,'betweenness')
    print('b_df,done')
    
    # 计算邻近中心性（Closeness Centrality）
    closeness_centrality = nx.closeness_centrality(G)
    c_df = centrality2dataframe(closeness_centrality,'closeness')
    print('c_df,done')

    # 计算特征向量中心性(Eigenvector_centrality)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    e_df = centrality2dataframe(eigenvector_centrality,'eigenvector')
    
    multicentrality_df = d_df.merge(b_df, on='mergedroad_index').merge(c_df, on='mergedroad_index').merge(e_df, on='mergedroad_index')
    multicentrality_df_withgeometry = pd.merge(multicentrality_df,merged_road,left_on='mergedroad_index',right_on='index')
    multicentrality_df_withgeometry.drop(columns='index',inplace=True)
    print('centrality_calculation_end')
    return multicentrality_df_withgeometry

#
def mergedroad2graph(merged_road,config):
    paired_road = []
    for index,row in merged_road.iterrows():
        print(f'running {index/len(merged_road)*100}%.')
        intersected_road = merged_road[row['geometry'].intersects(merged_road.geometry)]
        if len(intersected_road)>0:
            paired_road.extend([[index,i]for i in list(intersected_road['index']) if i != index] )
    
    graph = np.array(paired_road)
    save_graph = pd.DataFrame(graph,columns=['node1','node2'])
    save_graph.to_csv(config['path']['road_feature_gephi_path2'],index=False)
    
    return graph

#
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
    # merged road转图
    merged_road_graphed = mergedroad2graph(merged_road,config)
    print(merged_road_graphed.shape)
    # 计算多重中心性
    multicentrality_df = centrality_calculation(merged_road_graphed,merged_road,config)
    
    #将merged road的centrality index赋予单独街道段
    road_features = pair_road(road,multicentrality_df)
    
    with open(config['path']['road_feature_path'],'wb') as f:
        pickle.dump(road_features,f)
        
    road_features.to_file(config['path']['road_feature_shp_path'], encoding="utf-8")
    
    return road_features


