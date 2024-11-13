
import yaml
import pickle
import argparse
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import SGD,Adam,AdamW
from torch.nn import BCELoss,CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score, cohen_kappa_score

#Mine packages 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from checkpoint_dataset import CustomDataset

from checkpoint_models import QBresNet50,QBresnext50,QBVGG19,VIT,SwinTransformer


parser = argparse.ArgumentParser(description='An example')
parser.add_argument(
    "--modelconfig",
    default="checkpoint_models/config/checkpoint_config.yaml")

args = parser.parse_args()

print('parser done./n')

def _get_dataset_amount(dataset):
    total_amount = [label for label in dataset]
    label_0 = [zero['labels'] for zero in total_amount if zero['labels']==0]
    return {'total_amount':len(total_amount),'label_0':len(label_0),'labels_1':len(total_amount)-len(label_0)}

def get_param(layer):
    param_list = []
    for param in layer.parameters(): 
        param_list.append([param,param.grad])

    return param_list

def seed_worker(worker_id):
    np.random.seed(3407)
    torch.manual_seed(3407)

def makedataloader(config):
    
    with open(config['make_dataloader']['data_path'],"rb") as f:
        dataframe = pickle.load(f)
        
    dataset = CustomDataset(dataframe,augm = True)

    train_size = int(len(dataset)*0.7)
    eval_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(3407)
    train_dataset,eval_dataset = random_split(dataset,[train_size,eval_size], generator=generator)
    
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
        
    return graph_data

def choosing_model(config,device):
    
    # model
    if 'resnet50' in config['checkpoint_model']['name']:
        model = QBresNet50()
        # model = param_frozen(model,config)
        model.to(device)
        print('resnet50 loaded.')    
        
    elif 'resnext50' in config['checkpoint_model']['name']:
        model = QBresnext50()
        # model = param_frozen(model,config)
        model.to(device)
        print('resnext50 loaded.')    

    elif 'vgg19' in config['checkpoint_model']['name']:
        model = QBVGG19()
        # model = param_frozen(model,config)
        model.to(device)
        print('VGG19 loaded.')    
    
    elif 'vit' in config['checkpoint_model']['name']:
        model = VIT()
        # model = param_frozen(model,config)
        model.to(device)    
        print('VIT loaded.')    
    
    elif 'swintransformer' in config['checkpoint_model']['name']:
        model = SwinTransformer()
        model.to(device)    
        print('swintransformer loaded.')    
    
    elif 'Model_CrossAttention_VIT_GCN' in config['checkpoint_model']['name']:
        model = Model_CrossAttention_VIT_GCN(config)
        for param in model.pretrain_model.parameters():
            param.requires_grad = False
        model.to(device)
        print('Model_CrossAttention_VIT_GCN loaded.')
        
    # criterion
    criterion = nn.CrossEntropyLoss()
    
    print('{}_model loaded.'.format(config['checkpoint_model']['name']))
    
    #optimizer
    optimizer = Adam(model.parameters(),**config['choosing_model']['optimizer']['args'])

    lr_decay = CosineAnnealingLR(optimizer, **config['choosing_model']['CosineAnnealingLR'])
    
    return model,criterion,optimizer,lr_decay


def train(config,device,model,train_dataloader,graph_data,spatial_img,criterion,optimizer):

    running_loss = 0.0
    
    for batch in train_dataloader:
        
        img_scale = config['checkpoint_model']['img_scale']
        img = batch[img_scale].to(device)

        labels = batch["labels"].squeeze(1).long().to(device)
        
        optimizer.zero_grad()

        if 'Model_CrossAttention_VIT_GCN' in config['checkpoint_model']['name']:
            img1 = batch["img_s1"].to(device)
            img2 = batch["img_s2"].to(device)
            index = batch["roadidx"].to(device)
            spatial_img = spatial_img.to(device)
            graph_data = graph_data.to(device)
            output = model(img1,img2,spatial_img,index,graph_data)
        
        else:
            output = model(img)

        loss = criterion(output,labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()

    epoch_loss = running_loss/len(train_dataloader)

    return epoch_loss

def eval(config,device,model,eval_dataloader,graph_data,spatial_img,criterion):

    with torch.no_grad():
        predict_label = []
        true_label = []
        running_loss = 0.0
        
        for batch in eval_dataloader:
            
            img_scale = config['checkpoint_model']['img_scale']
            img = batch[img_scale].to(device)

            labels = batch["labels"].squeeze(1).long().to(device)
            if 'Model_CrossAttention_VIT_GCN' in config['checkpoint_model']['name']:
                img1 = batch["img_s1"].to(device)
                img2 = batch["img_s2"].to(device)
                index = batch["roadidx"].to(device)
                spatial_img = spatial_img.to(device)
                graph_data = graph_data.to(device)
                output = model(img1,img2,spatial_img,index,graph_data)
            else:
                output = model(img)
            
            true_label.append(labels.cpu().detach().numpy())
            predict_label.append(output.cpu().detach().numpy())            

            loss = criterion(output,labels)

            running_loss += loss.detach().item()

        eval_epoch_loss = running_loss/len(eval_dataloader)

    return predict_label,true_label,eval_epoch_loss


def evaluation(predict_label,true_label):

    all_true_label = []
    all_predict_label = []
    
    for item in range(len(true_label)):
        
        true = true_label[item]
        all_true_label.extend(true)
        predictions = np.argmax(predict_label[item], axis=1)
        all_predict_label.extend(predictions)

    oa = accuracy_score(all_true_label, all_predict_label)
    precision = precision_score(all_true_label, all_predict_label, zero_division=0)
    recall = recall_score(all_true_label, all_predict_label)
    f1 = f1_score(all_true_label, all_predict_label)
    kappa = cohen_kappa_score(all_true_label, all_predict_label)
    fpr, tpr, thresholds = roc_curve(all_true_label, all_predict_label)

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

    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(config['save_path']['loss_png_path'] + config['save_path']['experiment_id'])


def main(args):
    
    with open(args.modelconfig, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sp_img_path = 'crossattentionmodel0418/data/spatial_based_data_scale02_237.pickle'
    with open(sp_img_path,'rb') as spfile:
        spatial_img = pickle.load(spfile)
    train_dataloader,eval_dataloader = makedataloader(config)

    graph_data = streetnetwork(config)
    model,criterion,optimizer,lr_decay = choosing_model(config,device)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    train_loss = []
    eval_loss = []
    best_f1 = 0.0
    
    for epoch in range(config['train']['epoch_num']):
        
        model.train()
        train_epoch_loss = train(config,device,model,train_dataloader,graph_data,spatial_img,criterion,optimizer)       
        
        train_loss.append(train_epoch_loss)
        print("train {}_loss is:{}".format(epoch+1,train_epoch_loss))
        
        lr_decay.step()
        print("Learning rate:", optimizer.param_groups[0]['lr'])
        
        model.eval()
        predict_label,true_label,eval_epoch_loss = eval(config,device,model,eval_dataloader,graph_data,spatial_img,criterion)
        eval_loss.append(eval_epoch_loss)
        
        eval_matrix = evaluation(predict_label,true_label)
        
        print(f'eval_loss_is:{eval_epoch_loss}')
        # print('model_{}_evaluation_matrix is {}.'.format(config['checkpoint_model']['name'],eval_matrix))        
        
        if eval_matrix['Evaluation_indicators']['F1']>best_f1:
            best_f1 = eval_matrix['Evaluation_indicators']['F1']
            # torch.save(model.state_dict(),'crossattentionmodel0418/weight/' +'epoch_{}_bestmodel'.format(epoch) + '.pth')
            torch.save(model.state_dict(),f"checkpoint_models/weight/{config['checkpoint_model']['name']} bestmodel_weight.pth")
            print(f'***best_eval_F1 is {best_f1},save model.***')
            print(eval_matrix['Evaluation_indicators'])
    
    # with open('recover_yuan_road/model/checkpoint_models/infer_result','wb') as file:
    #     pickle.dump({'train_loss':train_loss,'eval_loss':eval_loss},file)
        
    
    print('checkpoint done.')


if __name__ =="__main__":

    args = parser.parse_args()

    main(args)
    