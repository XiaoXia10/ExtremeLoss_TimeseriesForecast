
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from os.path import join
import argparse
from data_scaler import destandardize_pred, load_loader_data

def get_metrics(args):
    loader = args.loader
    forecast = args.seq_length_x
    
    if args.scale_data==True:
        yhat, realy = destandardize_pred(args.df_path, args.save_dir, args.data_dir, args.loader)
    else:
        yhat, realy = load_loader_data(args.save_dir, args.data_dir, args.loader)
    
    vars()[loader+'_RMSE'] = root_mean_squared_error(realy, yhat)
    vars()[loader+'_R2'] = r2_score(realy, yhat)
    vars()[loader+'_MAPE'] = mean_absolute_percentage_error(realy, yhat)
    vars()[loader+'_MAE'] = mean_absolute_error(realy, yhat)
        
    
    f = open(join(args.save_dir, args.metric_name), 'w')
    f.write("Loader, Forecast, RMSE (l/s), R2, MAPE, MAE (l/s) \n" )
 
    f.write(str(loader)+",")
    f.write(str(forecast)+",")
    f.write(str(vars()[loader+'_RMSE'])+",")
    f.write(str(vars()[loader+'_R2'])+",")
    f.write(str(vars()[loader+'_MAPE'])+",")
    f.write(str(vars()[loader+'_MAE'])+",")
    f.write("\n") 
    f.close()
        
    print(loader, "\n")
    print("RMSE:")
    print(vars()[loader+'_RMSE'])
    print("R2:")
    print(vars()[loader+'_R2'])
    print("MAPE:")
    print(vars()[loader+'_MAPE'])
    print("MAE:")
    print(vars()[loader+'_MAE'], "\n")


if __name__ == "__main__":
    
    seq_length_x = 3
    seq_length_y = 3
    shift = 3
    loss = "extreme"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--df_path", type=str, default="data", help="df path",)
    
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--save_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"\experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
 
    parser.add_argument("--metric_name", type=str, default="GRU_model_metrics.csv", help="metric file name",)
    parser.add_argument("--loader", type=str, default="test", help="Which dataset to loader, loader = train, val, test",)
    
    parser.add_argument("--scale_data", type=bool, default=True, help="Data scaling",)
    
    args = parser.parse_args()
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
        
    get_metrics(args)
