from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, GRU
from tensorflow.keras import initializers, callbacks

# from extreme_loss import extreme_value_loss

def extreme_value_loss(y_true, y_pred, alpha=2.0):
    """
    Extreme Value Loss Function, focusing on extreme/outlier events in predictions.
    
    Parameters:
    y_true (tensor): True values.
    y_pred (tensor): Predicted values.
    alpha (float): Coefficient for standard deviation to define outliers.
    beta (float): Weight multiplier for high extreme values.
    gamma (float): Weight multiplier for low extreme values.
    
    Returns:
    loss (tensor): Computed loss value focusing on outlier events.
    """
    
    # Standard deviation and mean based on true values
    mean, variance = tf.nn.moments(y_true, axes=[0,1,2])
    std_dev = tf.sqrt(variance)
 
    # Define extremes: values beyond 'alpha' standard deviations from the mean
    extreme_mask_pos = tf.abs(y_true - mean) > alpha * std_dev
    extreme_mask_neg = tf.abs(y_true - mean) < alpha * std_dev

    # Calculate absolute errors
    errors = tf.abs(y_true - y_pred)
    # errors = tf.sqrt(tf.square(y_true - y_pred))
    
    
    dat_min = tf.reduce_min(y_true)
    dat_max = tf.reduce_max(y_true)
    
    beta = tf.abs(dat_max-mean)/std_dev
    gamma = tf.abs(mean-dat_min)/std_dev
    
    # Apply weights to errors based on whether the data is an outlier or not
    weights= tf.where(extreme_mask_pos, beta, 1.0) * tf.where(extreme_mask_neg, gamma, 1.0)
    # weights = tf.where(extreme_mask_pos, beta, tf.where(extreme_mask_neg, gamma, 1.0))
    
    weighted_errors = weights * errors

    # Mean error to form the loss
    loss = tf.reduce_mean(weighted_errors)
    
    return loss

def auto_encoder_gru(train_x, train_y, args):

    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = GRU(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h = encoder(encoder_inputs)

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_lstm = GRU(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=state_h)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    
    if args.extreme_loss == True:
        print("Extreme Loss function is used")
        model.compile(optimizer=adam_optimizer, loss=extreme_value_loss)
    else: 
        print("MAE Loss function is used")
        model.compile(optimizer=adam_optimizer, loss="mae")
    
    return model








