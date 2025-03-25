# src/evaluator.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from src.model import get_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def evaluate_model(model, X, y, threshold=0.3):
    predictions = model.predict(X)
    predictions_binary = (predictions > threshold).astype(int)
    accuracy = accuracy_score(y, predictions_binary)
    f1 = f1_score(y, predictions_binary)
    precision = precision_score(y, predictions_binary, zero_division=0)
    recall = recall_score(y, predictions_binary)
    cm = confusion_matrix(y, predictions_binary)
    return accuracy, f1, precision, recall, cm, predictions

def fine_tune_model(model, X, y, epochs=5):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy')
    X = X[..., np.newaxis]
    y = y[..., np.newaxis]
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, class_weight=class_weight_dict)

def plot_roc_curve(y_true, y_pred, filename='results/plots/roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_threshold_metrics(y_true, y_pred, thresholds=np.arange(0.1, 1.0, 0.1)):
    precisions, recalls, f1_scores, accuracies = [], [], [], []
    for threshold in thresholds:
        predictions_binary = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions_binary)
        precision = precision_score(y_true, predictions_binary, zero_division=0)
        recall = recall_score(y_true, predictions_binary)
        f1 = f1_score(y_true, predictions_binary)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='o')
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='o')
    plt.plot(thresholds, accuracies, label='Accuracy', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Classification Threshold')
    plt.legend()
    plt.grid()
    plt.savefig('results/plots/threshold_metrics.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)

    model = get_model()
    model.load_weights('models/meta_models/maml_model.weights.h5')

    # Load meta-test tasks
    tasks = np.load('data/processed/tasks.npy', allow_pickle=True)
    meta_accuracies, meta_f1s, meta_precisions, meta_recalls = [], [], [], []

    for task in tasks:
        X_test, y_test = task['test']
        X_test = X_test[..., np.newaxis]
        accuracy, f1, precision, recall, _, _ = evaluate_model(model, X_test, y_test)
        meta_accuracies.append(accuracy)
        meta_f1s.append(f1)
        meta_precisions.append(precision)
        meta_recalls.append(recall)

    # Load and preprocess the UNSW-NB15 test set
    test_data = pd.read_csv('data/raw/UNSW-NB15/UNSW_NB15_testing-set.csv', encoding='cp1252')
    if 'ï»¿id' in test_data.columns:
        test_data.rename(columns={'ï»¿id': 'id'}, inplace=True)

    # Drop unnecessary columns
    test_data.drop(columns=['id', 'attack_cat', 'proto', 'service', 'state'], axis=1, inplace=True)
    X_unsw = test_data.drop(columns=['label'])
    y_unsw = test_data['label'].values

    # Load selected features and scaler
    with open('data/processed/selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Keep only the selected features
    X_unsw = X_unsw[selected_features]
    X_unsw = pd.DataFrame(X_unsw, columns=selected_features)

    # Scale the test set
    X_unsw = scaler.transform(X_unsw)

    # Fine-tune the model on a subset of the UNSW-NB15 test data
    print("Fine-tuning model on UNSW-NB15 test data...")
    subset_size = int(0.1 * len(X_unsw))
    indices = np.random.choice(len(X_unsw), subset_size, replace=False)
    X_subset = X_unsw[indices]
    y_subset = y_unsw[indices]
    fine_tune_model(model, X_subset, y_subset, epochs=5)

    # Evaluate on the full UNSW-NB15 test set
    print("\nEvaluating on UNSW-NB15 testing set...")
    X_unsw = X_unsw[..., np.newaxis]
    accuracy, f1, precision, recall, cm, predictions = evaluate_model(model, X_unsw, y_unsw)
    print(f"UNSW-NB15 Test - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    print(f"\nMeta-Test - Average Accuracy: {np.mean(meta_accuracies):.4f}, Average F1-Score: {np.mean(meta_f1s):.4f}")
    print(f"Meta-Test - Average Precision: {np.mean(meta_precisions):.4f}, Average Recall: {np.mean(meta_recalls):.4f}")

    # Plot ROC curve
    plot_roc_curve(y_unsw, predictions)
    print("[ROC curve saved to results/plots/roc_curve.png]")

    # Plot metrics vs. threshold
    plot_threshold_metrics(y_unsw, predictions)
    print("[Threshold metrics plot saved to results/plots/threshold_metrics.png]")