# src/utils.py
import pickle
import struct
import socket

def send_data(socket, data):
    try:
        serialized_data = pickle.dumps(data)
        socket.sendall(struct.pack('!I', len(serialized_data)))
        socket.sendall(serialized_data)
    except (OSError, BrokenPipeError, ConnectionAbortedError) as e:
        print(f"Error sending data: {e}")
        raise

def receive_data(socket):
    try:
        # Receive the size of the data (4 bytes)
        data_size_bytes = socket.recv(4)
        if len(data_size_bytes) < 4:
            if len(data_size_bytes) == 0:
                return None  # Connection closed
            raise ValueError("Received incomplete data size")
        
        data_size = struct.unpack('!I', data_size_bytes)[0]
        if data_size == 0:
            return None
        
        # Receive the actual data
        received_data = b""
        while len(received_data) < data_size:
            packet = socket.recv(data_size - len(received_data))
            if not packet:
                raise ValueError("Connection closed while receiving data")
            received_data += packet
        
        return pickle.loads(received_data)
    except (OSError, ValueError, struct.error, ConnectionAbortedError) as e:
        print(f"Error receiving data: {e}")
        return None

# Note: This file handles socket communication on port 5000 (set in server.py).
# Prometheus metrics are served on ports 8000-8003, avoiding conflict with this port.