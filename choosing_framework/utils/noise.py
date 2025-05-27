import numpy as np

def introduce_label_noise(labels, noise_level):
    num_samples = len(labels)
    num_flips = int(noise_level * num_samples)
    noisy_indices = np.random.choice(num_samples, num_flips, replace=False)
    labels_copy = labels.copy()  # Criar uma cópia dos rótulos para não modificar os originais
    labels_copy.iloc[noisy_indices] = 1 - labels_copy.iloc[noisy_indices]  # Trocar os rótulos de algumas amostras
    
    return labels_copy, noisy_indices



def calculate_metrics(n_correct_hard, n_flipped, n_hard):
    recall = n_correct_hard / n_flipped if n_flipped > 0 else 0
    precision = n_correct_hard / n_hard if n_hard > 0 else 0
    f1_score = (2 * precision * recall )/ (precision + recall) if (precision + recall) > 0 else 0
    return f1_score



