# src/server.py
import sys
import os
import socket
from prometheus_client import Gauge, start_http_server
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
import yaml
from src.model import get_model
from src.meta_trainer import MetaTrainer
from src.utils import send_data, receive_data

# Define Prometheus metrics
global_loss = Gauge('global_model_loss', 'Loss of the global model after aggregation')
global_accuracy = Gauge('global_model_accuracy', 'Accuracy of the global model')
meta_loss_metric = Gauge('meta_loss', 'Meta loss from meta-update across tasks')

class Server:
    def __init__(self, host='localhost', port=5000, num_clients=3, metrics_port=8000):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.metrics_port = metrics_port  # Port for Prometheus metrics
        self.model = get_model()
        # Compile the model to enable evaluation
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
        print(f"Server model weights: {len(self.model.get_weights())}")
        self.model.summary()
        self.trainer = MetaTrainer(self.model)
        with open('configs/paths.yaml', 'r') as f:
            self.config_paths = yaml.safe_load(f)
        
        # Start HTTP server for Prometheus metrics
        start_http_server(self.metrics_port)
        print(f"Metrics server started at http://localhost:{self.metrics_port}")

    def evaluate_model(self, tasks):
        """Evaluate the global model on validation data from tasks."""
        # Check if tasks array is empty or the first task lacks 'test'
        if len(tasks) == 0 or 'test' not in tasks[0]:
            print("Warning: No valid test data in tasks. Using fallback values.")
            return 0.0, 0.0  # Fallback values if no test data
        validation_data = tasks[0]['test']
        # Unpack tuple (x_test, y_test) directly
        try:
            x_test, y_test = validation_data  # Expecting a tuple of (x_test, y_test)
            # Ensure correct shape for the model (input shape: (samples, 20, 1), output: (samples, 1))
            x_test = x_test[..., np.newaxis] if x_test.ndim == 2 else x_test
            y_test = y_test[..., np.newaxis] if y_test.ndim == 1 else y_test
            loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            return loss, accuracy
        except (ValueError, TypeError) as e:
            print(f"Evaluation error: {e}. Using fallback values. Check tasks[0]['test'] structure.")
            return 0.0, 0.0

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
                
                # Perform meta-update
                meta_loss = self.trainer.meta_update(avg_weights, tasks)
                print(f"Iteration {iteration + 1}/100 - Meta Loss: {meta_loss:.4f}")
                
                # Update model weights
                self.model.set_weights(avg_weights)
                
                # Evaluate the global model and update Prometheus metrics
                loss, accuracy = self.evaluate_model(tasks)
                global_loss.set(loss)
                global_accuracy.set(accuracy)
                meta_loss_metric.set(meta_loss)  # Add meta loss to Prometheus
                print(f"Iteration {iteration + 1}/100 - Global Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
                # Send updated weights to clients
                new_weights = self.model.get_weights()
                for conn in clients:
                    send_data(conn, new_weights)
                
                # Small delay to allow metrics to stabilize
                time.sleep(1)
            
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