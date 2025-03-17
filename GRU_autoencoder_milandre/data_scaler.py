import pandas as pd
from os.path import join
import numpy as np

def _get_stats(df_path):
        
    df = pd.read_csv(join(df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
        
    return std, mean


def standardize_df(df_path, loader):
    
    if loader == "test":
        df = pd.read_csv(join(df_path,"test_data.csv"), parse_dates=True, index_col=0)
    else:
        df = pd.read_csv(join(df_path,"train_val_data.csv"), parse_dates=True, index_col=0)
                
    std, mean =_get_stats(df_path)

    df= (df-mean.values)/std.values
    
    return df

 
def _load_loader_data(output_dir, loader):
       
    yhat = pd.read_csv(join(output_dir, loader+"_predy.csv"))
    realy = pd.read_csv(join(output_dir, loader+"_realy.csv"))
        
    return yhat, realy
    
def destandardize_pred(df_path, output_dir, time_dir, loader):
    
    yhat, realy =_load_loader_data(output_dir, loader)   
    std, mean = _get_stats(df_path)
    
    yhat = (yhat*std.values)+mean.values
    realy= (realy*std.values)+mean.values
    
    # # This time is only for testing dataset. Comment this out if you want training or validation dataset.
    # if loader == "test":
        
    #     time = np.load(join(time_dir, loader+"_time.npy"), allow_pickle=True) #
    #     time = pd.to_datetime(time.flatten())
           
    #     yhat.index = time
    #     realy.index = time
       
    #     return yhat, realy
    
    # else:
    #     return yhat, realy
    
    time = np.load(join(time_dir, loader+"_time.npy"), allow_pickle=True) #
    time = pd.to_datetime(time.flatten())
           
    yhat.index = time
    realy.index = time
       
    return yhat, realy
        