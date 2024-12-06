import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

pred_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results_val' #directory defined in "predict_validation.py"

performance_detection = pd.read_csv(os.path.join(pred_dir, 'metrics_golgi_detection.csv'), sep=';')
performance_detection = performance_detection.groupby(['gth'], as_index=False).agg({"TPR": np.mean, "FPR": np.mean})

fpr_values = performance_detection['FPR'].to_numpy()
tpr_values = performance_detection['TPR'].to_numpy()
thresholds = performance_detection['gth'].to_numpy()

def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

print('threshold best: {}'.format(cutoff_youdens_j(fpr_values, tpr_values, thresholds)))
