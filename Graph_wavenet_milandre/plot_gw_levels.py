
import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred

def plot_one_loader_segment(args):
    names = args.list_names
    
    yhat, realy = destandardize_pred(args.df_path, args.data_dir, args.time_dir, args.loader)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(20, 20), sharex=True,)
    ax1.plot(realy.index, realy.iloc[:,0], label="Measured Milandrine", color="r", linewidth=3)
    ax1.plot(yhat.index, yhat.iloc[:,0], label= "Predicted - "+loss+ " loss", color = "g", linewidth=3)
    ax1.set_title("GWN - "+ " Forecast Length " + str(forecast), fontsize = 35)
    ax1.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax2.plot(realy.index, realy.iloc[:,1], label="Measured Bâme", color="r", linewidth=3)
    ax2.plot(yhat.index, yhat.iloc[:,1], label="Predicted - "+loss+ " loss", color = "g", linewidth=3)
    ax2.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax3.plot(realy.index, realy.iloc[:,2], label="Measured Saivu", color="r", linewidth=3)
    ax3.plot(yhat.index, yhat.iloc[:,2], label="Predicted - "+loss+ " loss", color = "g", linewidth=3)
    ax3.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax4.plot(realy.index, realy.iloc[:,3], label="Measured Font", color="r", linewidth=3)
    ax4.plot(yhat.index, yhat.iloc[:,3], label="Predicted - "+loss+ " loss", color = "g", linewidth=3)
    ax4.set_xlabel("Date", fontsize = 25)
    ax4.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax1.tick_params(labelsize=25)
    ax1.legend(fontsize = 25)
    ax2.tick_params(labelsize=25)
    ax2.legend(fontsize = 25)
    ax3.tick_params(labelsize=25)
    ax3.legend(fontsize = 25)
    ax4.tick_params(labelsize=25)
    ax4.legend(fontsize = 25)
    # plt.savefig(r"G:\My Drive\Extreme_Loss_Function_Manuscript\figures\GWN_model\resampled_"+str(timestep)+"\GWN_forecast"+str(forecast)+"_"+loss+".png")

    
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
    path = r"G:\My Drive\Extreme_Loss_Function_Manuscript\figures\Milandrine_karst\GWN\resampled_"+str(timestep)
    
    yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
    yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
    
    fig, axes = plt.subplots(2,2, figsize=(20, 20))
    
    axes[0,0].plot(realy.iloc[:,0], yhat_mae.iloc[:,0], "o", color="blue", label="Milandrine - MAE Loss")
    axes[0,0].plot(realy.iloc[:,0], yhat_ext.iloc[:,0],  "*", color="orange", label="Milandrine - EXT Loss")
    axes[0,0].plot([0,realy.iloc[:,0].max()], [0,realy.iloc[:,0].max()], "k--", linewidth = 3)
    axes[0,0].set_xlabel("Measured (l/s)", fontsize = 25)
    axes[0,0].set_ylabel("Predicted (l/s)", fontsize = 25)
    axes[0,0].set(xlim=(0,realy.iloc[:,0].max()), ylim=(0,realy.iloc[:,0].max()))
    
    axes[0,1].plot(realy.iloc[:,1], yhat_mae.iloc[:,1], "o",  color="blue", label="Bâme - MAE Loss")
    axes[0,1].plot(realy.iloc[:,1], yhat_ext.iloc[:,1],  "*", color="orange", label="Bâme - EXT Loss")
    axes[0,1].plot([0,realy.iloc[:,1].max()], [0,realy.iloc[:,1].max()], "k--", linewidth = 3)
    axes[0,1].set_xlabel("Measured (l/s)", fontsize = 25)
    axes[0,1].set_ylabel("Predicted (l/s)", fontsize = 25)
    axes[0,1].set(xlim=(0,realy.iloc[:,1].max()), ylim=(0,realy.iloc[:,1].max()))
    
    axes[1,0].plot(realy.iloc[:,2], yhat_mae.iloc[:,2], "o",  color="blue", label="Saivu - MAE Loss")
    axes[1,0].plot(realy.iloc[:,2], yhat_ext.iloc[:,2],  "*", color="orange",label="Saivu - EXT Loss")
    axes[1,0].plot([0,realy.iloc[:,2].max()], [0,realy.iloc[:,2].max()], "k--", linewidth = 3)
    axes[1,0].set_xlabel("Measured (l/s)", fontsize = 25)
    axes[1,0].set_ylabel("Predicted (l/s)", fontsize = 25)
    axes[1,0].set(xlim=(0,realy.iloc[:,2].max()), ylim=(0,realy.iloc[:,2].max()))
    
    axes[1,1].plot(realy.iloc[:,3], yhat_mae.iloc[:,3], "o",  color="blue", label="Font - MAE Loss")
    axes[1,1].plot(realy.iloc[:,3], yhat_ext.iloc[:,3],  "*", color="orange", label="Font - EXT Loss")
    axes[1,1].plot([0,realy.iloc[:,3].max()], [0,realy.iloc[:,3].max()], "k--", linewidth = 3)
    axes[1,1].set_xlabel("Measured (l/s)", fontsize = 25)
    axes[1,1].set_ylabel("Predicted (l/s)", fontsize = 25)
    axes[1,1].set(xlim=(0,realy.iloc[:,3].max()), ylim=(0,realy.iloc[:,3].max()))
    
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
    
    path = r"G:\My Drive\Extreme_Loss_Function_Manuscript\figures\Milandrine_karst\GWN\resampled_"+str(timestep)
    
    yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
    yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(20, 20), sharex=True,)

    ax1.plot(realy.iloc[:,0],  "o--", color = "dimgray", label="Milandrine - Measured", linewidth = 3)
    ax1.plot(yhat_ext.iloc[:,0], "orange", label="Milandrine - EXT Loss", linewidth = 3)
    ax1.plot(yhat_mae.iloc[:,0], "blue", label="Milandrine - MAE Loss", linewidth = 3)
    ax1.set_ylabel("Discharge (l/s)", fontsize = 25)
    ax1.set_title("GWN - "+ " Forecast Length " + str(forecast), fontsize = 35)
    
    ax2.plot(realy.iloc[:,1], "o--", color = "dimgray", label="Bâme - Measured", linewidth = 3)
    ax2.plot(yhat_ext.iloc[:,1], "orange", label="Bâme - EXT Loss", linewidth = 3)
    ax2.plot(yhat_mae.iloc[:,1], "blue", label="Bâme - MAE Loss", linewidth = 3)
    ax2.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax3.plot(realy.iloc[:,2], "o--", color = "dimgray", label="Saivu - Measured", linewidth = 3)
    ax3.plot(yhat_ext.iloc[:,2], "orange", label="Saivu - EXT Loss", linewidth = 3)
    ax3.plot(yhat_mae.iloc[:,2], "blue", label="Saivu - MAE Loss", linewidth = 3)
    ax3.set_ylabel("Discharge (l/s)", fontsize = 25)
    
    ax4.plot(realy.iloc[:,3], "o--", color = "dimgray", label="Font - Measured", linewidth = 3)
    ax4.plot(yhat_ext.iloc[:,3], "orange", label="Font - EXT Loss", linewidth = 3)
    ax4.plot(yhat_mae.iloc[:,3], "blue", label="Font - MAE Loss", linewidth = 3)
    ax4.set_ylabel("Discharge (l/s)", fontsize = 25)
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

def generate_plot(args):
    
    # if args.one_loader == True:
    #     plot_one_loader_segment(args)
    
    # else:
    #     plot_entire_timerseries(args)
        
    cross_plot(args)
    plot_compare(args)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    forecast = 6
    loss ="mae" # extreme or mae

    
    timestep = "H" # H or 4H
    
    parser.add_argument("--data_dir", type=str, default="data_"+str(timestep)+"/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help="Model predicted data directory.")
    parser.add_argument("--time_dir", type=str, default="data_"+str(timestep)+"/GWN_"+str(forecast), help="Directory for time.",)

    parser.add_argument("--data_dir_ext", type=str, default="data_"+str(timestep)+"/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_extreme", help="Model predicted data directory.")
    parser.add_argument("--data_dir_mae", type=str, default="data_"+str(timestep)+"/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_mae", help="Model predicted data directory.")

    parser.add_argument("--df_path", type=str, default="data_"+str(timestep)+"", help="Training and testing data directory.",)
    
    parser.add_argument("--list_names", type=list, default=["Milandrine","Bâme","Saivu","Font" ], help="List of names of the monitoring stations",) #Keep double quotes
    
    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type",)
    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test",)
    
    args = parser.parse_args()

    generate_plot(args)