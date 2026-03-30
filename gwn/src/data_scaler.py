# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:36:45 2025

@author: Xiao Xia Liang
"""

import pandas as pd
from os.path import join
import numpy as np
import argparse
import os

# class data_scaler():
        
#     def __init__(self, df_path, output_dir, time_dir, loader):
        
#         df_path = self.df_path 
#         output_dir = self.output_dir 
#         time_dir = self.time_dir
#         loader = self.loader
    
#     def _get_stats(self):
        
#         df = pd.read_csv(join(self.df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
#         # Milamont is the flow measure of l/s
#         # This needs to scaled by itself
#         milandrine_std = df["Milandrine"].std()
#         milandrine_mean = df["Milandrine"].mean()
        
#         # These are measured in discharge m^3/s
#         temp = pd.concat([df.Bame, df.Saivu, df.Font])
#         std = temp.std()
#         mean = temp.mean()
        
#         return milandrine_std, milandrine_mean, std, mean
    
#     def _load_loader_data(self):
       
#         yhat = pd.read_csv(join(self.output_dir, self.loader+"_predy.csv"))
#         realy = pd.read_csv(join(self.output_dir, self.loader+"_realy.csv"))
        
#         return yhat, realy
    
    
#     def standardize_df(self):
        
#         if self.loader == "test":
#             df = pd.read_csv(join(self.df_path,"test_data.csv"), parse_dates=True, index_col=0)
#         else:
#             df = pd.read_csv(join(self.df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
            
#         milandrine_std, milandrine_mean, std, mean = data_scaler._get_stats()

#         df.Bame = (df.Bame-mean)/std
#         df.Saivu = (df.Saivu-mean)/std
#         df.Font = (df.Font-mean)/std
#         df.Milandrine = (df.milandrine-milandrine_mean)/milandrine_std
    
#         return df


#     def destandardize_pred(self):
    
#         if self.loader == "test":
#             df = pd.read_csv(join(self.df_path,"test_data.csv"), parse_dates=True, index_col=0)
#         else:
#             df = pd.read_csv(join(self.df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
    
    
#         yhat, realy = data_scaler._load_loader_data()

#         milandrine_std, milandrine_mean, std, mean = data_scaler._get_stats()

#         yhat.iloc[:,0] = (yhat.iloc[:,0]*milandrine_std)+milandrine_mean
#         realy.iloc[:,0] = (realy.iloc[:,0]*milandrine_std)+milandrine_mean
            
#         yhat.iloc[:,1:] = (yhat.iloc[:,1:]*std)+mean
#         realy.iloc[:,1:] = (realy.iloc[:,1:]*std)+mean
    
#         time = np.load(join(self.time_dir, self.loader+"_time.npy"), allow_pickle=True)
#         time = pd.to_datetime(time.flatten())
           
#         yhat.index = time
#         realy.index = time
       
#         return yhat, realy

def _get_stats(df_path):
        
    df = pd.read_csv(join(df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
        
    return std, mean

def _load_loader_data(output_dir, loader):
       
    yhat = pd.read_csv(join(output_dir, loader+"_predy.csv"))
    realy = pd.read_csv(join(output_dir, loader+"_realy.csv"))
        
    return yhat, realy
    

def standardize_df(df_path, loader):
    
    if loader == "test":
        df = pd.read_csv(join(df_path,"test_data.csv"), parse_dates=True, index_col=0)
    else:
        df = pd.read_csv(join(df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
                
    std, mean =_get_stats(df_path)

    df= (df-mean.values)/std.values
    
    return df


def log_df(df_path, loader):
    
    if loader == "test":
        df = pd.read_csv(join(df_path,"test_data.csv"), parse_dates=True, index_col=0)
    else:
        df = pd.read_csv(join(df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
    
    df = np.log10(df)
    
    return df

def exp10_pred(df_path, output_dir, time_dir, loader):
    
    yhat, realy =_load_loader_data(output_dir, loader)   
    std, mean = _get_stats(df_path)
    
    # yhat = (yhat*std.values)+mean.values
    # realy = (realy*std.values)+mean.values
    yhat = np.power(10.0, yhat)
    realy = np.power(10.0, realy)
    
    # This time is only for testing dataset. Comment this out if you want training or validation dataset.
    if loader == "test":
        
        time = np.load(join(time_dir, loader+"_time.npy"), allow_pickle=True) #
        time = pd.to_datetime(time.flatten())
           
        yhat.index = time
        realy.index = time
       
        return yhat, realy
    
    else:
        return yhat, realy
    
 

def destandardize_pred(df_path, output_dir, time_dir, loader):
    
    yhat, realy =_load_loader_data(output_dir, loader)   
    std, mean = _get_stats(df_path)
    
    yhat = (yhat*std.values)+mean.values
    realy= (realy*std.values)+mean.values
    
    # This time is only for testing dataset. Comment this out if you want training or validation dataset.
    if loader == "test":
        
        time = np.load(join(time_dir, loader+"_time.npy"), allow_pickle=True) #
        time = pd.to_datetime(time.flatten())
           
        yhat.index = time
        realy.index = time
       
        return yhat, realy
    
    else:
        return yhat, realy
    
        
    
    