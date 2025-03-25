# src/meta_trainer.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class MetaTrainer:
    def __init__(self, model, meta_lr=0.000001):
        self.model = model
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr, clipnorm=1.0)

    def meta_update(self, avg_weights, tasks):
        # Set the averaged weights from FedAvg
        self.model.set_weights(avg_weights)

        meta_loss = 0
        for task in tasks:
            X_test, y_test = task['test']
            X_test = X_test[..., np.newaxis]
            y_test = tf.expand_dims(y_test, axis=-1)
            
            # Convert y_test to float32 to match the type of weight_pos and weight_neg
            y_test = tf.cast(y_test, dtype=tf.float32)
            
            # Flatten y_test and convert to NumPy array for sklearn
            y_test_flat = tf.reshape(y_test, [-1]).numpy()
            
            # Compute class weights for the test set
            class_weights = compute_class_weight('balanced', classes=np.unique(y_test_flat), y=y_test_flat)
            class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
            
            with tf.GradientTape() as tape:
                predictions = self.model(X_test, training=True)
                # Apply class weights to the loss
                loss = tf.keras.losses.binary_crossentropy(y_test, predictions)
                weight_pos = tf.constant(class_weight_dict[1], dtype=tf.float32)
                weight_neg = tf.constant(class_weight_dict[0], dtype=tf.float32)
                weighted_loss = tf.reduce_mean(loss * (y_test * weight_pos + (1 - y_test) * weight_neg))
            
            meta_loss += weighted_loss

            gradients = tape.gradient(weighted_loss, self.model.trainable_variables)
            self.meta_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        meta_loss = meta_loss / len(tasks)
        return meta_loss.numpy()

if __name__ == "__main__":
    from src.model import get_model
    model = get_model()
    trainer = MetaTrainer(model)
    print("MetaTrainer initialized successfully.")