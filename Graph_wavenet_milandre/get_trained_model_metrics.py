
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
from os.path import join
import numpy as np
import argparse
import os
from data_scaler import destandardize_pred

def get_metrics(args):
    loader = args.loader
    forecast = args.forecast
 
 
    f = open(join(args.data_dir, args.metric_name), 'w')
    f.write("Loader, Forecast, RMSE (l/s), R2, MAPE, MAE (l/s) \n" )

    yhat, realy = destandardize_pred(args.df_path, args.data_dir, args.time_dir, args.loader)
    
    vars()[loader+'_RMSE'] = root_mean_squared_error(realy, yhat)
    vars()[loader+'_R2'] = r2_score(realy, yhat)
    vars()[loader+'_MAPE'] = mean_absolute_percentage_error(realy, yhat)
    vars()[loader+'_MAE'] = mean_absolute_error(realy, yhat)
        
    f.write(str(loader)+",")
    f.write(str(forecast)+",")
    f.write(str(vars()[loader+'_RMSE'])+",")
    f.write(str(vars()[loader+'_R2'])+",")
    f.write(str(vars()[loader+'_MAPE'])+",")
    f.write(str(vars()[loader+'_MAE'])+",")
    f.write("\n") 
        
    print(loader, "\n")
    print("RMSE:")
    print(vars()[loader+'_RMSE'])
    print("R2:")
    print(vars()[loader+'_R2'])
    print("MAPE:")
    print(vars()[loader+'_MAPE'])
    print("MAE:")
    print(vars()[loader+'_MAE'], "\n")

    f.close()

if __name__ == "__main__":
    
    loss = "extreme"
    forecast = 9
    timestep = "4H"
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--forecast", type=int, default=forecast, help="metric file name",)
    parser.add_argument("--df_path", type=str, default="data_"+timestep, help="df path",)
    parser.add_argument("--data_dir", type=str, default="data_"+timestep+"/GWN_"+str(forecast)+"/experiment_"+str(forecast)+"_"+loss, help="Data directory")
    parser.add_argument("--time_dir", type=str, default="data_"+timestep+"/GWN_"+str(forecast), help="Time directory")

    parser.add_argument("--metric_name", type=str, default="GWN_model_metrics.csv", help="metric file name",)
    parser.add_argument("--loader", type=str, default="test", help="Which dataset to loader, loader = train, val, test",)
    
    args = parser.parse_args()
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
        
    get_metrics(args)

