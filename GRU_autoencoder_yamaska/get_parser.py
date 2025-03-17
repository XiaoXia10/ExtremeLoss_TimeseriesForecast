
import argparse

def get_shared_arg_parser():
    
    seq_length_x = 3
    seq_length_y = 3
    shift = 3
    loss = "extreme"
    
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--extreme_loss', type=str, default=True, help='If True use extreme loss function, if False use MAE')
        
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') 
    parser.add_argument('--latent_dim', type=int, default=500, help='Latent dimension') 
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout') 
    parser.add_argument('--recurrent_dropout', type=float, default=0.7, help='Recurrent dropout') 
        
   
    # parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--patience', type=int, default=20, help='quit if no improvement after this many iterations')

    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--data_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift), help="df for testing.",)
    parser.add_argument("--save_dir", type=str, default="data\GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"\experiment_GRU"+str(seq_length_x)+str(seq_length_y)+str(shift)+"_"+loss, help="Save Path")
    parser.add_argument("--save_model_name", type=str, default="GRU_best_model.h5", help="Best saved model name")
   
    return parser
