# IoT Meta-Learning IDS

## Overview
The **IoT Meta-Learning IDS** project develops an Intrusion Detection System (IDS) for IoT networks using **meta-learning** (MAML) and **federated learning** (FL). The model, a **hybrid Fish Swarm ConvNet**, is trained on the **UNSW-NB15 dataset** for binary classification (normal vs. attack traffic). The approach ensures privacy-preserving, adaptive, and scalable intrusion detection.

## Features
- **Federated Learning**: Clients train locally, preserving data privacy.
- **Meta-Learning (MAML)**: Enhances fast adaptation to new attack patterns.
- **Hybrid Fish Swarm ConvNet**: Improves detection accuracy.
- **High Performance**: Achieves **91.84% accuracy**, **0.9287 F1-score**, and **0.98 AUC**.

## Project Structure
```
iot_meta_learning_ids-1/
│── configs/
│   └── paths.yaml             # Stores file paths for datasets, models, and results
│── data/
│   ├── raw/                   # UNSW-NB15 dataset
│   ├── processed/             # Preprocessed data (tasks, test sets)
│── src/
│   ├── clients/               # Simulated IoT clients (client1.py, client2.py, ...)
│   ├── model.py               # Fish Swarm ConvNet model architecture
│   ├── data_loader.py         # Data preprocessing and task generation
│   ├── meta_trainer.py        # MAML meta-training process
│   ├── server.py              # Federated learning server
│   ├── evaluator.py           # Model evaluation and visualization
│   ├── utils.py               # Helper functions for client-server communication
│── results/
│   └── plots/                 # ROC curve, precision-recall graphs
```

## Workflow
1. **Data Preprocessing (`data_loader.py`)**  
   - Load and preprocess UNSW-NB15 dataset  
   - Select top 20 features, scale data, generate meta-learning tasks  

2. **Federated Learning (`server.py`, `clients/`)**  
   - Clients train on local tasks  
   - Server aggregates updates (FedAvg) and updates global model  

3. **Meta-Learning (`meta_trainer.py`)**  
   - Train Fish Swarm ConvNet using MAML for fast adaptation  

4. **Evaluation (`evaluator.py`)**  
   - Fine-tune and test the model  
   - Compute metrics, generate ROC curves, recommend threshold  

## Installation
```bash
git clone https://github.com/your-repo/iot_meta_learning_ids-1.git
cd iot_meta_learning_ids-1
pip install -r requirements.txt
```

## Running the Project
1. **Preprocess Data**  
   ```bash
   python src/data_loader.py
   ```
2. **Start Federated Learning**  
   ```bash
   python src/server.py &  # Run server
   python src/clients/client1.py &  # Run clients
   ```
3. **Run Meta-Training**  
   ```bash
   python src/meta_trainer.py
   ```
4. **Evaluate Model**  
   ```bash
   python src/evaluator.py
   ```

## Results
- **91.84% Accuracy**, **0.9287 F1-score**, **0.98 AUC**
- Recommended threshold **0.4** for best precision-recall balance

## License
MIT License © 2025 Manasi Pandit

