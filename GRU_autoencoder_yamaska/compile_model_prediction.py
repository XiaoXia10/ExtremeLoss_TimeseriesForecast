
import tensorflow as tf
from model import auto_encoder_gru
from get_parser  import get_shared_arg_parser
import pandas as pd
import numpy as np
from os.path import join


def main(args):
    
    checkpoint_path = join(args.save_dir, args.save_model_name)

    train_x = np.load(join(args.data_dir, "x_train.npy"))
    train_y = np.load(join(args.data_dir, "y_train.npy"))

    model = auto_encoder_gru(train_x, train_y, args)

    model.load_weights(checkpoint_path) 

    val_x = np.load(join(args.data_dir, "x_val.npy"))
    val_y = np.load(join(args.data_dir, "y_val.npy"))
    test_x = np.load(join(args.data_dir, "x_test.npy"))
    test_y = np.load(join(args.data_dir, "y_test.npy"))

    pred_test = model.predict(test_x)
    pred_train = model.predict(train_x)
    pred_val = model.predict(val_x)
    
    loss_train = model.evaluate(train_x, train_y, verbose=2)
    print("Restored model, train loss: {:5.2f}".format(loss_train))

    loss_val = model.evaluate(val_x, val_y, verbose=2)
    print("Restored model, val loss: {:5.2f}".format(loss_val))

    loss_test = model.evaluate(test_x, test_y, verbose=2)
    print("Restored model, test loss: {:5.2f}".format(loss_test))
    
    pred_test = np.reshape(pred_test, (pred_test.shape[0]*pred_test.shape[1], pred_test.shape[2]))
    test_y = np.reshape(test_y, (test_y.shape[0]*test_y.shape[1], test_y.shape[2]))
    
    pred_val = np.reshape(pred_val, (pred_val.shape[0]*pred_val.shape[1], pred_val.shape[2]))
    val_y = np.reshape(val_y, (val_y.shape[0]*val_y.shape[1], val_y.shape[2]))
    
    pred_train = np.reshape(pred_train, (pred_train.shape[0]*pred_train.shape[1], pred_train.shape[2]))
    train_y = np.reshape(train_y, (train_y.shape[0]*train_y.shape[1], train_y.shape[2]))
    
    pred_test = pd.DataFrame(pred_test)
    test_y = pd.DataFrame(test_y)
    
    pred_val = pd.DataFrame(pred_val)
    val_y = pd.DataFrame(val_y)
    
    pred_train = pd.DataFrame(pred_train)
    train_y = pd.DataFrame(train_y)
    
    train_y.to_csv(join(args.save_dir, "train_realy.csv"), index =False)
    val_y.to_csv(join(args.save_dir, "val_realy.csv"), index =False)
    test_y.to_csv(join(args.save_dir, "test_realy.csv"), index =False)
    
    pred_train.to_csv(join(args.save_dir, "train_predy.csv"), index =False)
    pred_val.to_csv(join(args.save_dir, "val_predy.csv"), index =False)
    pred_test.to_csv(join(args.save_dir, "test_predy.csv"), index =False)

if __name__ == "__main__":
    parser = get_shared_arg_parser()
    args = parser.parse_args()
    main(args)
    