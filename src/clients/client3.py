# src/clients/client3.py
import sys
import os
import socket

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
from src.model import get_model
from src.utils import send_data, receive_data
from sklearn.utils.class_weight import compute_class_weight

class Client:
    def __init__(self, client_id, host='localhost', port=5000):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.model = get_model()
        print(f"Client {self.client_id} model weights: {len(self.model.get_weights())}")
        self.tasks = np.load('data/processed/tasks.npy', allow_pickle=True)[client_id - 1]

    def local_train(self):
        X_train, y_train = self.tasks['train']
        X_train = X_train[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy')
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, class_weight=class_weight_dict)
        
        return self.model.get_weights(), len(X_train)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            print(f"Client {self.client_id} connected to server")
            while True:
                weights, num_samples = self.local_train()
                try:
                    send_data(s, (weights, num_samples))
                except (OSError, ConnectionAbortedError, ConnectionResetError) as e:
                    print(f"Client {self.client_id} connection closed by server: {e}")
                    break
                
                new_weights = receive_data(s)
                if new_weights is None:
                    print(f"Client {self.client_id} received stop signal or connection closed by server")
                    break
                
                self.model.set_weights(new_weights)

if __name__ == "__main__":
    client = Client(client_id=3)
    client.run()