import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

pred_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results_val' #directory defined in "predict_validation.py"

performance_detection = pd.read_csv(os.path.join(pred_dir, 'metrics_nuclei_detection.csv'), sep=';')
performance_detection = performance_detection.groupby(['nth'], as_index=False).agg({"TPR": np.mean, "FPR": np.mean})

fpr_values = performance_detection['FPR'].to_numpy()
tpr_values = performance_detection['TPR'].to_numpy()
thresholds = performance_detection['nth'].to_numpy()

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

print('thresholds best: {}'.format(cutoff_youdens_j(fpr_values, tpr_values, thresholds)))


# Calculate Youden's J-Statistic for each threshold
j_statistic = [tpr - fpr for tpr, fpr in zip(tpr_values, fpr_values)]

# Find the index of the threshold with the highest J-Statistic
best_threshold_index = np.argmax(j_statistic)

# Retrieve the best threshold
best_threshold = thresholds[best_threshold_index]

# Print the best threshold and corresponding TPR and FPR
print("Best Threshold:", best_threshold)
print("Corresponding TPR:", tpr_values[best_threshold_index])
print("Corresponding FPR:", fpr_values[best_threshold_index])