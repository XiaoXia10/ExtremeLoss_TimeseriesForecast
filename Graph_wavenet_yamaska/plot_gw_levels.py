
import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred, load_loader_data

def plot_one_loader_segment(args):
    
    path = r""
    names = np.load(args.name_file)
    
    if args.scale_data == True:
        
        yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
    
    else:
        yhat_ext, realy = load_loader_data(args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = load_loader_data(args.data_dir_mae, args.time_dir, args.loader)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(20, 20), sharex=True,)
    ax1.plot(realy.index, realy.iloc[:,4], 'o--', label="Mea. Well " + str(names[4]), color="dimgray", linewidth=3)
    ax1.plot(yhat_ext.index, yhat_ext.iloc[:,4], label= "Pred. - EXT loss", color = "orange", linewidth=3)
    ax1.plot(yhat_mae.index, yhat_mae.iloc[:,4], label= "Pred. - MAE loss", color = "blue", linewidth=3)
    ax1.set_title("GWN - "+ " Forecast Length " + str(forecast), fontsize = 35)
    ax1.set_ylabel("GW Level (masl)", fontsize = 25)
    
    ax2.plot(realy.index, realy.iloc[:,6], 'o--', label="Mea. Well " + str(names[6]), color="dimgray", linewidth=3)
    ax2.plot(yhat_ext.index, yhat_ext.iloc[:,6], label= "Pred. - EXT loss", color = "orange", linewidth=3)
    ax2.plot(yhat_mae.index, yhat_mae.iloc[:,6], label= "Pred. - MAE loss", color = "blue", linewidth=3)
    ax2.set_ylabel("GW Level (masl)", fontsize = 25)
    
    ax3.plot(realy.index, realy.iloc[:,8], 'o--', label="Mea. Well " + str(names[8]), color="dimgray", linewidth=3)
    ax3.plot(yhat_ext.index, yhat_ext.iloc[:,8], label= "Pred. - EXT loss", color = "orange", linewidth=3)
    ax3.plot(yhat_mae.index, yhat_mae.iloc[:,8], label= "Pred. - MAE loss", color = "blue", linewidth=3)
    ax3.set_ylabel("GW Level (masl)", fontsize = 25)
    
    ax4.plot(realy.index, realy.iloc[:,9], 'o--', label="Mea. Well " + str(names[9]), color="dimgray", linewidth=3)
    ax4.plot(yhat_ext.index, yhat_ext.iloc[:,9], label= "Pred. - EXT loss", color = "orange", linewidth=3)
    ax4.plot(yhat_mae.index, yhat_mae.iloc[:,9], label= "Pred. - MAE loss", color = "blue", linewidth=3)
    ax4.set_ylabel("GW Level (masl)", fontsize = 25)
    ax4.set_xlabel("Date", fontsize = 25)

    ax1.tick_params(labelsize=25)
    ax1.legend(fontsize = 25)
    ax2.tick_params(labelsize=25)
    ax2.legend(fontsize = 25)
    ax3.tick_params(labelsize=25)
    ax3.legend(fontsize = 25)
    ax4.tick_params(labelsize=25)
    ax4.legend(fontsize = 25)
    plt.savefig(join(path, "GWN_forecast"+str(forecast)+".png"))
    

    
def plot_entire_timerseries(args):
    
    station = 1 # 0 to 3 total of 4 stations, "Milandrine","Bame","Saivu","Font"  
    names = args.list_names
    
    loaders = ["train","val", "test"]
    df_all_yhat = pd.DataFrame()
    df_all_realy = pd.DataFrame()
    
    plt.figure(figsize=(25,5))
    for loader in loaders:
        yhat, realy = destandardize_pred(args.df_path, args.data_dir, args.time_dir, args.loader)
        
        df_all_yhat = pd.concat([df_all_yhat, yhat])
        df_all_realy = pd.concat([df_all_realy, realy])
        
        plt.plot(realy.index, realy.iloc[:,station], color="r", linewidth=2)
        plt.plot(yhat.index, yhat.iloc[:,station], label="Pred_"+loader, linewidth=2)
    
    plt.title(str(names[station]), fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Discharge (l/s)', fontsize=20) # Milandrine is measured at l/s, the other springs are measured in m^3/s
    plt.legend(fontsize=20)
    plt.axis("equal")  
    # for i in range(0, df_all_yhat.shape[1]):
        
    #     plt.figure(figsize=(25,5))
    #     plt.plot(df_all_realy.index, df_all_realy.iloc[:,i], label="True")
    #     plt.plot(df_all_yhat.index, df_all_yhat.iloc[:,i], label="Pred")
    #     plt.title(names[i], fontsize=20)
    #     plt.xlabel('Date', fontsize=20)
    #     plt.ylabel('Discharge', fontsize=25)
    #     plt.legend()

def cross_plot(args):
    
    
    path = r""
    names = np.load(args.name_file)
    
    yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
    yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
     
    fig, axes = plt.subplots(2,2, figsize=(20, 20))
     
    axes[0,0].plot(realy.iloc[:,4], yhat_mae.iloc[:,4], "o", color = "blue", label= "Well " +str(names[4])+" - MAE Loss")
    axes[0,0].plot(realy.iloc[:,4], yhat_ext.iloc[:,4],  "*", color = "orange", label="Well " +str(names[4])+" - EXT Loss")
    axes[0,0].plot([0,realy.iloc[:,4].max()], [0,realy.iloc[:,4].max()], "k--", linewidth = 3)
    axes[0,0].set_xlabel("Measured (masl)", fontsize = 25)
    axes[0,0].set_ylabel("Predicted (masl)", fontsize = 25)
    axes[0,0].set(xlim=(realy.iloc[:,4].min(),realy.iloc[:,4].max()), ylim=(realy.iloc[:,4].min(),realy.iloc[:,4].max()))
     
    axes[0,1].plot(realy.iloc[:,6], yhat_mae.iloc[:,6], "o", color = "blue", label= "Well " +str(names[6])+" - MAE Loss")
    axes[0,1].plot(realy.iloc[:,6], yhat_ext.iloc[:,6],  "*", color = "orange", label="Well " +str(names[6])+" - EXT Loss")
    axes[0,1].plot([0,realy.iloc[:,6].max()], [0,realy.iloc[:,6].max()], "k--", linewidth = 3)
    axes[0,1].set_xlabel("Measured (masl)", fontsize = 25)
    axes[0,1].set_ylabel("Predicted (masl)", fontsize = 25)
    axes[0,1].set(xlim=(realy.iloc[:,6].min(), realy.iloc[:,6].max()), ylim=(realy.iloc[:,6].min(), realy.iloc[:,6].max()))
     
    axes[1,0].plot(realy.iloc[:,8], yhat_mae.iloc[:,8], "o", color = "blue", label= "Well " +str(names[8])+" - MAE Loss")
    axes[1,0].plot(realy.iloc[:,8], yhat_ext.iloc[:,8],  "*", color = "orange", label="Well " +str(names[8])+" - EXT Loss")
    axes[1,0].plot([0,realy.iloc[:,8].max()], [0,realy.iloc[:,8].max()], "k--", linewidth = 3)
    axes[1,0].set_xlabel("Measured (masl)", fontsize = 25)
    axes[1,0].set_ylabel("Predicted (masl)", fontsize = 25)
    axes[1,0].set(xlim=(realy.iloc[:,8].min(),realy.iloc[:,8].max()), ylim=(realy.iloc[:,8].min(),realy.iloc[:,8].max()))
      
    axes[1,1].plot(realy.iloc[:,9], yhat_mae.iloc[:,9], "o", color = "blue", label= "Well " +str(names[9])+" - MAE Loss")
    axes[1,1].plot(realy.iloc[:,9], yhat_ext.iloc[:,9],  "*", color = "orange", label="Well " +str(names[9])+" - EXT Loss")
    axes[1,1].plot([0,realy.iloc[:,9].max()], [0,realy.iloc[:,9].max()], "k--", linewidth = 3)
    axes[1,1].set_xlabel("Measured (masl)", fontsize = 25)
    axes[1,1].set_ylabel("Predicted (masl)", fontsize = 25)
    axes[1,1].set(xlim=(realy.iloc[:,9].min(),realy.iloc[:,9].max()), ylim=(realy.iloc[:,9].min(),realy.iloc[:,9].max()))
     
    axes[0,0].tick_params(labelsize=25)
    axes[0,0].legend(fontsize = 25)
    axes[0,1].tick_params(labelsize=25)
    axes[0,1].legend(fontsize = 25)
    axes[1,0].tick_params(labelsize=25)
    axes[1,0].legend(fontsize = 25)
    axes[1,1].tick_params(labelsize=25)
    axes[1,1].legend(fontsize = 25)
    
    plt.suptitle("GWN - "+ " Forecast Length " + str(forecast), fontsize = 35)
    plt.tight_layout(pad=2.0)
    plt.savefig(join(path, "GWN_forecast"+str(forecast)+"_CP.png"))
    
def plot_compare(args):
    
    name = np.load(args.name_file)
    
    if args.scale_data == True:
        
        yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
    
    else:
        yhat_ext, realy = load_loader_data(args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = load_loader_data(args.data_dir_mae, args.time_dir, args.loader)
    
    
    path = r""
    
    for i in range(0, yhat_mae.shape[1]):
        plt.figure(figsize=(15,5))
        plt.plot(realy.index, realy.iloc[:,i], 'o--', label="Measured", color="dimgray", linewidth=1.5)
        plt.plot(yhat_ext.index, yhat_ext.iloc[:,i], label="EXT Loss", color = "orange", linewidth=3)
        plt.plot(yhat_mae.index, yhat_mae.iloc[:,i], label="MAE Loss", color = "blue", linewidth=3)
        plt.title("Well "+str(name[i])+ " - Forecast Length " + str(forecast), fontsize=25)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Water Level (MASL)', fontsize=20)
        plt.legend(fontsize=20)
        
        plt.savefig(join(path, str(name[i])+"ForecastLength"+str(forecast)+".png"))

def generate_plot(args):
    
    if args.one_loader == True:
        plot_one_loader_segment(args)
    
    else:
        plot_entire_timerseries(args)
        
    cross_plot(args)
    # plot_compare(args)
    
    
if __name__ == "__main__":
    
    forecast = 5
    loss ="extreme" # extreme or mae
    
    parser = argparse.ArgumentParser()

 
    
    parser.add_argument("--data_dir", type=str, default="data/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help="Model predicted data directory.")
    parser.add_argument("--time_dir", type=str, default="data/GWN_"+str(forecast), help="Directory for time.",)

    parser.add_argument("--data_dir_ext", type=str, default="data/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_extreme", help="Model predicted data directory.")
    parser.add_argument("--data_dir_mae", type=str, default="data/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_mae", help="Model predicted data directory.")

    parser.add_argument("--df_path", type=str, default="data", help="Training and testing data directory.",)
    
    parser.add_argument("--name_file", type=str, default="well_numbers.npy", help="List of names of the monitoring stations",) #Keep double quotes or s**t
    
    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type",)
    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test.",)
    parser.add_argument("--scale_data", type=bool, default=True, help="Data scaling",)
    
    args = parser.parse_args()

generate_plot(args)