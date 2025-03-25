# src/data_loader.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_unsw_nb15_data(data_path):
    # Check if preprocessed data exists
    processed_file = 'data/processed/UNSW_NB15_training_processed.csv'
    if os.path.exists(processed_file):
        df = pd.read_csv(processed_file)
        selected_features = df.drop(columns=['label']).columns
        X = df.drop(columns=['label']).values
        y = df['label'].values
        # Load the scaler
        with open('data/processed/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return X, y, selected_features

    # If preprocessed data doesn't exist, process the raw data
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.dropna(inplace=True)

    # Fix malformed 'id' column name
    if 'ï»¿id' in df.columns:
        df.rename(columns={'ï»¿id': 'id'}, inplace=True)

    # Drop unnecessary columns
    drop_list = ['id', 'attack_cat', 'proto', 'service', 'state']
    df.drop(drop_list, axis=1, inplace=True)

    # Ensure 'label' exists
    if 'label' not in df.columns:
        df['label'] = df.get('Label', df.get('attack_cat', 0)).apply(lambda x: 1 if x != 0 else 0)

    # Feature selection
    X = df.drop(columns=['label'])
    y = df['label'].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    selected_features = X.columns[indices]
    print(f"Selected features: {selected_features}")

    X = X[selected_features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Save the scaler and selected features
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('data/processed/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)

    # Save the processed data
    processed_data = pd.DataFrame(X, columns=selected_features)
    processed_data['label'] = y
    processed_data.to_csv('data/processed/UNSW_NB15_training_processed.csv', index=False)

    return X, y, selected_features

def generate_tasks(X, y, num_tasks=5, train_size=200, test_size=100):
    tasks = []
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    samples_per_class_train = train_size // 2
    samples_per_class_test = test_size // 2

    for _ in range(num_tasks):
        pos_train_idx = np.random.choice(pos_indices, samples_per_class_train, replace=False)
        neg_train_idx = np.random.choice(neg_indices, samples_per_class_train, replace=False)
        train_indices = np.concatenate([pos_train_idx, neg_train_idx])

        remaining_pos_indices = np.setdiff1d(pos_indices, pos_train_idx)
        remaining_neg_indices = np.setdiff1d(neg_indices, neg_train_idx)

        pos_test_idx = np.random.choice(remaining_pos_indices, samples_per_class_test, replace=False)
        neg_test_idx = np.random.choice(remaining_neg_indices, samples_per_class_test, replace=False)
        test_indices = np.concatenate([pos_test_idx, neg_test_idx])

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        task = {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
        tasks.append(task)

    return tasks

if __name__ == "__main__":
    with open('configs/paths.yaml', 'r') as f:
        config_paths = yaml.safe_load(f)

    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw/UNSW-NB15', exist_ok=True)

    X, y, selected_features = load_unsw_nb15_data(config_paths['datasets']['unsw_nb15'])

    tasks = generate_tasks(X, y)
    np.save(config_paths['processed']['tasks'], tasks)

    test_sets = {'unsw_nb15': pd.DataFrame(X, columns=selected_features)}
    test_sets['unsw_nb15']['label'] = y
    with open('data/processed/test_sets.pkl', 'wb') as f:
        pickle.dump(test_sets, f)

    print(f"Generated {len(tasks)} tasks for simulated IoT devices.")