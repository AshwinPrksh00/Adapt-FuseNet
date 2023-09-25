import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import *
from tensorflow.keras.layers import *

def Conv3DBlock(idx, filters, kernels, act,  input_vector):
    x = Conv3D(filters, kernel_size=kernels, strides=2, activation=act)(input_vector)
    x = BatchNormalization(name=f'BNLayer3D_{idx}')(x)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same', name=f'Maxpool3D_{idx}')(x)
    return x

def ConvLSTM2DBlock(idx, filters, kernels, input_vector):
    ind = idx
    x = ConvLSTM2D(filters, kernel_size=kernels, return_sequences=True, name=f'Conv2DBlock_{ind}')(input_vector)
    x = MaxPooling3D(pool_size=3, strides=2, padding='same', name=f'Maxpool3D_{ind}')(x)
    x = BatchNormalization(name=f'BNLayer_{ind}')(x)
    return x

def FCBlock(n_class, n_units, drop, act, input_vector):
    assert type(n_units) == int, f"Type of units should be of {int}"
    assert type(act) == str, f"Type of activations should be of {str}"
    assert type(drop) == float, f"Type of dropout should be of {float}"
    
    if input_vector is None:
        raise Exception('Input Vector cannot be None')
    
    x = Dense(units=n_units, activation=act)(input_vector)
    x = Dropout(drop)(x)
    
    output_vector = Dense(units=n_class, activation='softmax')(x)
    
    return output_vector
    