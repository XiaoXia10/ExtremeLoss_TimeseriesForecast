import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, GRU
from tensorflow.keras import initializers, callbacks
import numpy as np

def extreme_loss(y_true, y_pred, alpha=2.5):
    """
    Extreme Value Loss Function, focusing on extreme events in predictions.
    
    Parameters:
    y_true (tensor): True values.
    y_pred (tensor): Predicted values.
    alpha (float): Coefficient for standard deviation to define extremes.
    beta (float): Weight multiplier for high extreme values.
    gamma (float): Weight multiplier for low extreme values.
    
    Returns:
    loss (tensor): Computed loss value focusing on extreme events.
    """
    
    # Standard deviation and mean based on true values
    mean, variance = tf.nn.moments(y_true, axes=[0,1,2])
    std_dev = tf.sqrt(variance)
 
    dat_min = tf.reduce_min(y_true)
    dat_max = tf.reduce_max(y_true)
    
    beta = tf.abs(dat_max-mean)/std_dev
    gamma = tf.abs(mean-dat_min)/std_dev

    # Define extremes: values beyond 'alpha' standard deviations from the mean
    extreme_mask_pos = tf.abs(y_true - mean) > alpha * std_dev
    extreme_mask_neg = tf.abs(y_true - mean) < alpha * std_dev

    # Calculate absolute errors
    errors = tf.abs(y_true - y_pred)
    # errors = tf.sqrt(tf.square(y_true - y_pred))
    # errors = tf.square(y_true - y_pred)

    # Apply weights to errors based on whether the data is extreme or not
    # weights= tf.where(extreme_mask_pos, beta, 1.0) * tf.where(extreme_mask_neg, gamma, 1.0)
    weights = tf.where(extreme_mask_pos, beta, tf.where(extreme_mask_neg, gamma, 1.0))
    
    weighted_errors = weights * errors

    # Mean error to form the loss
    loss = tf.reduce_mean(weighted_errors)
    
    return loss
