
import pandas as pd
import numpy as np
import argparse
from os.path import join
import os
from data_scaler import standardize_df

def sequence_data_preparation(args, df):
    
    seq_length_x, seq_length_y, shift = args.seq_length_x, args.seq_length_y, args.shift
    
    num_samples, num_nodes = df.shape
    data = df.to_numpy()
    
    df_time = df.index
    max_t = num_samples - (seq_length_y+shift)
    x, y, time_list = [], [], []
    
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x,:])
        y.append(total_window_len[seq_length_x:,:]) 
        
        time_window_len = df_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time = np.stack(time_list, axis=0)
    
    return x, y, time


def main(args):
    
        
    if args.scale_data == True:
        df = standardize_df(args.df_path, loader="train")
        df_test = standardize_df(args.df_path, loader="test")
    else:
        df = pd.read_csv(args.df_train, index_col=0, parse_dates=True)
        df_test = pd.read_csv(args.df_test, index_col=0, parse_dates=True)


    x, y, time = sequence_data_preparation(args, df)
    x_test, y_test, time_test= sequence_data_preparation(args, df_test)
    
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_percent)
    # num_val = num_samples - num_train

    x_train, y_train, time_train = x[:num_train], y[:num_train], time[:num_train]

    x_val, y_val, time_val = x[num_train:], y[num_train:], time[num_train:]
    
    np.save(join( args.output_dir, "x_train.npy"), x_train)
    np.save(join( args.output_dir, "y_train.npy"), y_train)
    np.save(join( args.output_dir, "x_val.npy"), x_val)
    np.save(join( args.output_dir, "y_val.npy"), y_val)
    np.save(join( args.output_dir, "x_test.npy"), x_test)
    np.save(join( args.output_dir, "y_test.npy"), y_test)
    np.save(join(args.output_dir, "test_time.npy"), time_test)
    np.save(join(args.output_dir, "val_time.npy"), time_val)
    np.save(join(args.output_dir, "train_time.npy"), time_train)

    return 0

if __name__ == "__main__":
    
    seq_length_x = 6
    seq_length_y = 6
    shift = 6
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--df_path", type=str, default="data", help="df path",)
    
    parser.add_argument("--df_train", type=str, default="data\train_val_data.csv", help="df for train and validation.",)
    parser.add_argument("--df_test", type=str, default="data\test_data.csv", help="df for testing.",)
    
    parser.add_argument("--output_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    
    parser.add_argument('--train_percent', type=float, default=0.8, help='The percentage of data used for model training')
    parser.add_argument("--scale_data", type=bool, default=True, help="Data scaling",)
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    main(args)