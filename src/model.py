# src/model.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import tensorflow as tf

def get_model(input_shape=(20, 1), num_classes=1):
    """
    Define a CNN model for intrusion detection with regularization.
    Args:
        input_shape: Shape of the input data (e.g., (20, 1) for 20 features with 1 channel).
        num_classes: Number of output classes (1 for binary classification).
    Returns:
        A TensorFlow model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', name='conv1'),
        tf.keras.layers.BatchNormalization(name='bn1'),
        tf.keras.layers.MaxPooling1D(pool_size=2, name='pool1'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name='conv2'),
        tf.keras.layers.BatchNormalization(name='bn2'),
        tf.keras.layers.MaxPooling1D(pool_size=2, name='pool2'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense1'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid', name='output')
    ], name='hybrid_ids')
    
    return model

if __name__ == "__main__":
    model = get_model()
    model.summary()