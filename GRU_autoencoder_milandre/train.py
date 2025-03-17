
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from model import auto_encoder_gru
from durbango import pickle_save
import argparse
from get_parser import get_shared_arg_parser

def train(args, **model_kwargs):
    
    train_x = np.load(join(args.data_dir, "x_train.npy"))
    train_y = np.load(join(args.data_dir, "y_train.npy"))
    val_x = np.load(join(args.data_dir, "x_val.npy"))
    val_y = np.load(join(args.data_dir, "y_val.npy"))
      
    model = auto_encoder_gru(train_x, train_y, args)
    
    model.summary()
        
    # Save the best trained model with the min val_error
    save_path = join(args.save_dir, args.save_model_name)
    

    # Regulate the model
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=20, 
                                             verbose=1, 
                                             mode='min')

    model_checkpoint =  callbacks.ModelCheckpoint(filepath=save_path, 
                                                  monitor='val_loss', 
                                                  save_best_only=True, 
                                                  verbose=1)

    list_callback = [early_stopping, model_checkpoint]

    # Train the model
    history = model.fit(x=train_x, 
                        y=train_y, 
                        epochs=args.epochs, 
                        batch_size= args.batch_size, 
                        validation_data=(val_x, val_y),
                        callbacks=list_callback,
                        verbose=2, 
                        shuffle=False)

    # plot history
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss', fontsize= 15)
    plt.ylabel('Loss', fontsize= 15)
    plt.xlabel('Epoch', fontsize= 15)
    plt.legend()

if __name__ == "__main__":
    
    parser = get_shared_arg_parser()
    args = parser.parse_args()
        
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    pickle_save(args, f'{args.save_dir}/args.pkl')
    
    train(args)
        


    