from sklearn.metrics import confusion_matrix, accuracy_score, \
precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize = (6,5)):
  df_cm = pd.DataFrame(
      confusion_matrix, index=class_names, columns=class_names,
  )
  fig = plt.figure(figsize=figsize)
  try:
      heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Reds')
  except ValueError:
      raise ValueError("Confusion matrix values must be integers.")
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
  plt.title('Confusion matrix', fontsize=25)
  plt.ylabel('True label', fontsize=17)
  plt.xlabel('Predicted label', fontsize=17)
  plt.show()
  return fig

def plot_f1_vs_thr(y_true, y_pred_prob):
  scores = []
  thr = np.linspace(0,1,100)
  for t in thr:
    y_pred = (y_pred_prob>=t).astype(int)
    f1 = f1_score(y_true, y_pred)
    scores.append(f1)
  ind = np.argmax(scores)
  print(f'Optimal threshold is {thr[ind]:.4f}, at f1 score of {scores[ind]:.4f}')
  plt.plot(thr,scores,ms=5,marker='.',linestyle='-')
  plt.xlim([0.3,0.7])
  plt.title('F1 score vs threshold')
  plt.xlabel('Threshold')
  plt.ylabel('F1 score')
  plt.grid(which='major', color='k')
  plt.grid(which='minor', linestyle='--')
  plt.minorticks_on()
  plt.show()
   


def evaluate_model(y_true, y_pred, y_pred_prob):
  '''
  y_true: a numpy array of true class label, containing 0 and 1
  y_pred: a numpy array of predicted class label, containing 0 and 1
  y_pred_prob: a numpy array of predicted probability of belonging to a class
  '''

  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  cm = np.array([tp,fn,fp,tn]).reshape(2,2)
  acc = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred_prob)

  print('Accuracy: %.3f'%acc)
  print('Precision: %.3f'%precision)
  print('Recall: %.3f'%recall)
  print('F1 Score: %.3f'%f1)
  print('AUC score: %.3f \n'%roc_auc)
  print_confusion_matrix(cm,[1,0])
  

def plot_roc(y_true, y_pred_prob):
  '''
  y_true: a numpy array of true class label, containing 0 and 1
  y_pred_prob: a numpy array of predicted probability of belonging to a class
  '''

  fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=None, drop_intermediate=False)
  roc_auc = roc_auc_score(y_true, y_pred_prob)
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(loc="lower right")
  plt.show()

