# src/server.py
import sys
import os
import socket

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
import yaml
from src.model import get_model
from src.meta_trainer import MetaTrainer
from src.utils import send_data, receive_data

class Server:
    def __init__(self, host='localhost', port=5000, num_clients=3):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.model = get_model()
        print(f"Server model weights: {len(self.model.get_weights())}")
        self.model.summary()
        self.trainer = MetaTrainer(self.model)
        with open('configs/paths.yaml', 'r') as f:
            self.config_paths = yaml.safe_load(f)

    def fed_avg(self, client_data):
        """
        Implement Federated Averaging (FedAvg) to aggregate client weights.
        Args:
            client_data: List of tuples (weights, num_samples) from each client.
        Returns:
            Aggregated weights.
        """
        total_samples = sum(num_samples for _, num_samples in client_data)
        if total_samples == 0:
            return None
        
        # Initialize aggregated weights
        avg_weights = [np.zeros_like(w) for w in client_data[0][0]]
        
        # Weighted average of client weights
        for weights, num_samples in client_data:
            weight_factor = num_samples / total_samples
            for i in range(len(weights)):
                avg_weights[i] += weights[i] * weight_factor
        
        return avg_weights

    def run(self):
        tasks = np.load('data/processed/tasks.npy', allow_pickle=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(self.num_clients)
            print("Server started...")
            print("Waiting for clients to connect...")
            clients = []
            for _ in range(self.num_clients):
                conn, addr = s.accept()
                clients.append(conn)
                print(f"Connected to client {len(clients)}")
            
            for iteration in range(100):  # Increased to 100 meta-iterations
                client_data = []
                for conn in clients:
                    data = receive_data(conn)
                    client_data.append(data)
                
                # Aggregate client weights using FedAvg
                avg_weights = self.fed_avg(client_data)
                if avg_weights is None:
                    print("No client data received.")
                    break
                
                meta_loss = self.trainer.meta_update(avg_weights, tasks)
                print(f"Iteration {iteration + 1}/100 - Meta Loss: {meta_loss:.4f}")
                
                new_weights = self.model.get_weights()
                for conn in clients:
                    send_data(conn, new_weights)
            
            for conn in clients:
                send_data(conn, None)
                conn.close()
        
        meta_model_path = self.config_paths['models']['meta_model']
        os.makedirs(os.path.dirname(meta_model_path), exist_ok=True)
        self.model.save_weights(meta_model_path)
        print(f"Meta-model weights saved to {meta_model_path}")
        
        print("Training completed.")

if __name__ == "__main__":
    server = Server()
    server.run()