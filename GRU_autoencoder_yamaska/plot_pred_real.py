
import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from data_scaler import destandardize_pred, load_loader_data


def plot_one_loader_segment(args):

    path = r"G:\My Drive\Extreme_Loss_Function_Manuscript\figures\Yamasha_wells\GRU"
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
    ax1.set_title("GRU - "+ " Forecast Length " + str(seq_length_x), fontsize = 35)
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
    plt.savefig(join(path, "GRU_forecast"+str(seq_length_x)+".png"))

def cross_plot(args):
    
    path = r""
    names = np.load(args.name_file)
    
    yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.data_dir, args.loader)
    yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.data_dir, args.loader)
     
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
    
    plt.suptitle("GRU - "+ " Forecast Length " + str(seq_length_x), fontsize = 35)
    plt.tight_layout(pad=2.0)
    plt.savefig(join(path, "GRU_forecast"+str(seq_length_x)+"_CP.png"))
    
def plot_compare(args):
    
    name = np.load(args.name_file)
    # print(name)
    
    if args.scale_data == True:
        
        yhat_ext, realy = destandardize_pred(args.df_path, args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = destandardize_pred(args.df_path, args.data_dir_mae, args.time_dir, args.loader)
    
    else:
        yhat_ext, realy = load_loader_data(args.data_dir_ext, args.time_dir, args.loader)
        yhat_mae, _ = load_loader_data(args.data_dir_mae, args.time_dir, args.loader)
        

    for i in range(0, yhat_mae.shape[1]):
        plt.figure(figsize=(15,5))
        plt.plot(realy.index, realy.iloc[:,i], 'o--', label="Measured", color="dimgray", linewidth=1.5)
        plt.plot(yhat_ext.index, yhat_ext.iloc[:,i], label="EXT Loss", color = "orange", linewidth=3)
        plt.plot(yhat_mae.index, yhat_mae.iloc[:,i], label="MAE Loss", color = "blue", linewidth=3)
        plt.title("Well "+str(name[i])+ " - Forecast Length " + str(seq_length_y), fontsize=25)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Water Level (MASL)', fontsize=20)
        plt.legend(fontsize=20)
     
    

def plot_entire_timerseries(args):
    names = args.name_file
    
    loaders = ["train","val", "test"]
    df_all_yhat = pd.DataFrame()
    df_all_realy = pd.DataFrame()
    
    plt.figure(figsize=(15,5))
    for loader in loaders:
        yhat, realy = destandardize_pred(args.df_path, args.save_dir, args.data_dir, args.loader)
        
        df_all_yhat = pd.concat([df_all_yhat, yhat])
        df_all_realy = pd.concat([df_all_realy, realy])
        
        plt.plot(realy.index, realy.iloc[:,2], label=loader, linewidth=2)
        # plt.plot(realy.index, realy.iloc[:,2], color="r", linewidth=2)
        # plt.plot(yhat.index, yhat.iloc[:,2], label="Pred_"+loader, linewidth=2)
    
    # plt.title("Font", fontsize=20)
    # plt.xlabel('Date', fontsize=15)
    plt.ylabel('Discharge (l/s)', fontsize=20)
    plt.legend(fontsize=20)
    
    # for i in range(0, df_all_yhat.shape[1]):
        
    #     plt.figure(figsize=(15,5))
    #     plt.plot(df_all_realy.index, df_all_realy.iloc[:,i], color = "r", label="True")
    #     plt.plot(df_all_yhat.index, df_all_yhat.iloc[:,i], label="Pred")
    #     plt.title(names[i], fontsize=18)
    #     plt.xlabel('Date', fontsize=15)
    #     plt.ylabel('MASL', fontsize=15)
    #     plt.legend()
        
def generate_plot(args):
    
    if args.one_loader == True:
        plot_one_loader_segment(args)
    
    else:
        plot_entire_timerseries(args)
    
    cross_plot(args)
    # plot_compare(args)
    
    
if __name__ == "__main__":
    
    seq_length_x = 5
    seq_length_y = 5
    shift = 5
    loss ="extreme"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--df_path", type=str, default="data", help="df path")
    
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.")
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.")
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--time_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Get time",)
    parser.add_argument("--save_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"\experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
    parser.add_argument("--save_model_name", type=str, default="GRU_best_model.h5", help="Best saved model name")
   
    parser.add_argument("--data_dir_ext", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"\experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_extreme", help="Model predicted data directory.")
    parser.add_argument("--data_dir_mae", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"\experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_mae", help="Model predicted data directory.")

    
    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type.")
    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test.")
    parser.add_argument("--scale_data", type=bool, default=True, help="Data scaling")
    
    parser.add_argument("--name_file", type=str, default="well_numbers.npy", help="Well numbers")
    
    args = parser.parse_args()

    generate_plot(args)