
import yaml
import random
import pickle
import argparse
import torch
import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD,Adam,AdamW
from torch.nn import BCELoss,CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score,cohen_kappa_score

#Mine packages 
from crossattention_dataset import CustomDataset
from crossattention_model import Model_CrossAttention_VIT_GCN,Concat,Addition,Scale1_GCN,Scale2_GCN,img1_only_model,img2_only_model,img1_img2_model,Road_Structure_only_model


#args.modelconfig
parser = argparse.ArgumentParser(description='An example')
parser.add_argument(
    "--modelconfig",
    default="crossattentionmodel0418/config/attentionmodel_config.yaml")

def seed_worker(worker_id):
    np.random.seed(3407)
    torch.manual_seed(3407)

def makedataloader(config):

    with open(config['make_dataloader']['data_path'],"rb") as f:
        dataframe = pickle.load(f)

    dataset = CustomDataset(dataframe,augm = True)

    train_size = int(len(dataset)*0.7)
    test_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(3407)
    train_dataset,eval_dataset = random_split(dataset,[train_size,test_size], generator=generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['make_dataloader']['batch_size'], shuffle=True, worker_init_fn=seed_worker)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['make_dataloader']['batch_size'], shuffle=True, worker_init_fn=seed_worker)
    return train_dataloader,eval_dataloader

def streetnetwork(config):
    #road
    with open(config['street_network']['road_path'],'rb') as f1:
        road_network = pickle.load(f1)
    
    #nodes feature
    with open(config['street_network']['nodes_features_path'],'rb') as f2:
        nodes_data = pickle.load(f2)
    
    #clean data
    nodes_data.drop(columns=['index','geometry'],inplace = True)

    #scaleing data
    nodes_data*10000
    for column in list(nodes_data.columns):
        nodes_data[column] = (nodes_data[column]-nodes_data[column].min())/(nodes_data[column].max()-nodes_data[column].min())

    #to tensor
    nodes_features = torch.tensor(nodes_data.values, dtype=torch.float32)

    mean_value = nodes_features[~torch.isnan(nodes_features)].mean()
    nodes_features = torch.where(torch.isnan(nodes_features), mean_value, nodes_features)

    #
    if config['street_network']['GCN']:

        edge_index = torch.tensor(road_network, dtype=torch.long).t()
        graph_data = Data(x=nodes_features, edge_index=edge_index)
        
    else:

        edge_index = torch.tensor(road_network, dtype=torch.long).t()
        after_cat = torch.cat((edge_index[0].unsqueeze(1).t(),edge_index[1].unsqueeze(1).t()),dim=1)
        maximum_node_index = max(list(set(after_cat.tolist()[0])))

        adj_matrix = torch.zeros(maximum_node_index+1,maximum_node_index+1)
        for col in tqdm(range(edge_index.shape[1])):
            adj_matrix[edge_index[0,col],edge_index[1,col]] = 1

        graph_data = {'nodes_feature':nodes_features,"adjacent_matrix":adj_matrix}
    
    return graph_data

def choosing_model(config,device):
    # model
    if config['choosing_model']['model']['name'] == 'Model_CrossAttention_VIT_GCN':
        model = Model_CrossAttention_VIT_GCN(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Model_CrossAttention_VIT_GCN model loaded.')
    
    # merging comparaion
    elif config['choosing_model']['model']['name'] == 'Concat':
        model = Concat(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Concat model loaded.')
                
    elif config['choosing_model']['model']['name'] == 'Addition':
        model = Addition(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Addition model loaded.')
    
    # ablition 
    elif config['choosing_model']['model']['name'] == 'Scale1_GCN':
        model = Scale1_GCN(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Scale1_GCN model loaded.')
    
    elif config['choosing_model']['model']['name'] == 'Scale2_GCN':
        model = Scale2_GCN(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Scale2_GCN model loaded.')
    
    elif config['choosing_model']['model']['name'] == 'img1_only_model':
        model = img1_only_model(config)
        model = param_frozen(model,config)
        model.to(device)
        print('img1_only_model model loaded.')

    elif config['choosing_model']['model']['name'] == 'img2_only_model':
        model = img2_only_model(config)
        model = param_frozen(model,config)
        model.to(device)
        print('img2_only_model model loaded.')

    elif config['choosing_model']['model']['name'] == 'img1_img2_model':
        model = img1_img2_model(config)
        model = param_frozen(model,config)
        model.to(device)
        print('img1_img2_model model loaded.')

    elif config['choosing_model']['model']['name'] == 'Road_Structure_only_model':
        model = Road_Structure_only_model(config)
        model = param_frozen(model,config)
        model.to(device)
        print('Road_Structure_only_model model loaded.')
    
    
    # optimizer
    Optimizer = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW}[config['choosing_model']['optimizer']['name']]
    optimizer = Optimizer(model.parameters(),**config['choosing_model']['optimizer']['args'])
    
    # criterion
    criterion = {'BCE':BCELoss,
                 "Crossentropyloss":CrossEntropyLoss}[config['choosing_model']['criterion']['name']]()
    
    # 
    lr_decay = CosineAnnealingLR(optimizer, **config['choosing_model']['CosineAnnealingLR'])

    return model,optimizer,criterion,lr_decay


def param_frozen(model,config):
    if config['choosing_model']['param']['load_param'] == 'pth':
        model_state_dict = model.state_dict()
        pretrain_model = torch.load(config['choosing_model']['param']['pretrain_weight'])
        
        #
        for name,params in pretrain_model.items():
            if name in model_state_dict.keys() and model_state_dict[name].shape == params.shape:
                model_state_dict[name] = params
            else:
                print(name)     
        
        #
        model.load_state_dict(model_state_dict, strict=False)
        
        #
        for name, param in model.named_parameters():
            if "pretrain" in name:
                param.requires_grad = False
    
    elif config['choosing_model']['param']['load_param'] == 'bin':
        
        for param in model.pretrain_model.parameters():
            param.requires_grad = False
        
    return model
                
def train(config,train_dataloader,graph_data,spatial_img,model,optimizer,criterion,device):

    running_loss = 0.0
    for batch in train_dataloader:
        
        if isinstance(batch,tuple):
            for items in range(len(batch)):
                if isinstance(batch[items],dict):
                    batch = batch[items]
                    
        img1 = batch["img_s1"].to(device)
        img2 = batch["img_s2"].to(device)
        spatial_img = spatial_img.to(device)
        index = batch["roadidx"].to(device)
        graph_data.to(device)
        
        labels = batch["labels"].squeeze(1).long().to(device)
        
        optimizer.zero_grad()
        result = model(img1,img2,spatial_img,index,graph_data)
        loss = criterion(result,labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.detach().item()
    
    epoch_loss = running_loss/len(train_dataloader)
    
    return epoch_loss

def eval(config,eval_dataloader,graph_data,spatial_img,model,optimizer,criterion,device):


    with torch.no_grad():
        predict_label = []
        true_label = []
        running_loss = 0.0
        for batch in eval_dataloader:

            if isinstance(batch,tuple):
                for items in range(len(batch)):
                    if isinstance(batch[items],dict):
                        batch = batch[items]

            img1 = batch["img_s1"].to(device)
            img2 = batch["img_s2"].to(device)
            spatial_img = spatial_img.to(device)
            index = batch["roadidx"].to(device)
            graph_data = graph_data.to(device)
            labels = batch["labels"].squeeze(1).long().to(device)

            true_label.append(labels.detach().cpu().numpy())
            
            result = model(img1,img2,spatial_img,index,graph_data)
        
            predict_label.append(result.detach().cpu().numpy())
            loss = criterion(result,labels)
            running_loss += loss.detach().item()
        epoch_loss = running_loss/len(eval_dataloader)
        
    return epoch_loss,predict_label,true_label

def evaluation(predict_label,true_label):
    predict_list = []
    true_list = []
    for count in range(len(predict_label)):

        predictions = np.argmax(predict_label[count], axis=1)
        predict_list.extend(predictions)
        true_list.extend(true_label[count]) 

    # 
    oa = accuracy_score(true_list, predict_list)
    precision = precision_score(true_list, predict_list)
    recall = recall_score(true_list, predict_list)
    f1 = f1_score(true_list, predict_list)
    kappa = cohen_kappa_score(true_list, predict_list)
    fpr, tpr, thresholds = roc_curve(true_list, predict_list)
    roc_auc = auc(fpr, tpr)
    
    eval_matrix = {'Evaluation_indicators':{'OA':oa,'Precision':precision,'Recall':recall,'F1':f1,'kappa':kappa},
            'ROC':{'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds,'roc_auc':roc_auc}}

    return eval_matrix

def save_result(path,data):
    with open(path,'wb') as f:
        pickle.dump(data,f)

def plot(config,train_loss,eval_loss):
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(eval_loss, label='Eval Loss', color='red')

    #
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #
    plt.savefig(config['save_path']['loss_png_path'])


def main(args):
    #
    with open(args.modelconfig, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('parse done.')
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407) 
    
    # 
    sp_img_path = 'crossattentionmodel0418/data/spatial_based_data_scale02_237.pickle'
    with open(sp_img_path,'rb') as spfile:
        spatial_img = pickle.load(spfile)
    
    #
    train_dataloader,eval_dataloader = makedataloader(config)

    print('dataloader done.')
    
    #
    graph_data = streetnetwork(config)
    model,optimizer,criterion,lr_decay = choosing_model(config,device)
    #
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    #
    epoch_lr_list = []
    eval_matrix_list = []
    train_losses = []
    eval_losses = []
    eval_best_loss = float('inf')
    early_stop = 0
    best_f1 = 0.0
    for epoch in range(config['train']['epoch_num']):
        model.train()
        #
        epoch_train_loss = train(config,train_dataloader,graph_data,spatial_img,model,optimizer,criterion,device)
        lr_decay.step()
        #
        train_losses.append(epoch_train_loss)
        print(f"       training {epoch} epoch loss is {epoch_train_loss}      ")
        model.eval()
        epoch_eval_loss,predict_label,true_label = eval(config,eval_dataloader,graph_data,spatial_img,model,optimizer,criterion,device)
        print(f"       evaluate {epoch} epoch loss is {epoch_eval_loss}      ")

        eval_losses.append(epoch_eval_loss)

        eval_matrix = evaluation(predict_label,true_label)
        eval_matrix_list.append(eval_matrix)
        
        #save model weight pth 
        if eval_matrix['Evaluation_indicators']['F1']>best_f1:
            best_f1 = eval_matrix['Evaluation_indicators']['F1']
            # torch.save(model.state_dict(),'crossattentionmodel0418/weight/' +'epoch_{}_bestmodel'.format(epoch) + '.pth')
            torch.save(model.state_dict(),'crossattentionmodel0418/weight/bestmodel_weight.pth')
            print(f'***best_eval_F1 is {best_f1},save model.***')
            print(eval_matrix['Evaluation_indicators'])
        
    return train_losses,eval_losses

if __name__ =="__main__":

    args = parser.parse_args()

    main(args)