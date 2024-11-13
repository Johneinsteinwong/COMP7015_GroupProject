from sklearn.metrics import confusion_matrix, accuracy_score, \
precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer 
from sklearn.base import BaseEstimator, TransformerMixin


def compute_features(df, n_neighbors=10):
    # Make sure not using label to compute features
    assert 'mortality' not in df.columns

    # Calculate Range
    df['Fraction inspired oxygen_range'] =  df['Fraction inspired oxygen_max'] -  df['Fraction inspired oxygen_min']
    df['Glucose_range'] =  df['Glucose_max'] -  df['Glucose_min']
    df['Heart Rate_range'] =  df['Heart Rate_max'] -  df['Heart Rate_min']
    df['Mean blood pressure_range'] =  df['Mean blood pressure_max'] -  df['Mean blood pressure_min']
    df['Diastolic blood pressure_range'] =  df['Diastolic blood pressure_max'] -  df['Diastolic blood pressure_min']
    df['Systolic blood pressure_range'] =  df['Systolic blood pressure_max'] -  df['Systolic blood pressure_min']
    df['Oxygen saturation_range'] =  df['Oxygen saturation_max'] -  df['Oxygen saturation_min']
    df['Respiratory rate_range'] =  df['Respiratory rate_max'] -  df['Respiratory rate_min']
    df['Temperature_range'] =  df['Temperature_max'] -  df['Temperature_min']
    df['Weight_range'] =  df['Weight_max'] -  df['Weight_min']
    df['pH_range'] =  df['pH_max'] -  df['pH_min']

    # Create Ratios
    df['Fraction inspired oxygen_min_to_mean'] = df['Fraction inspired oxygen_min'] / df['Fraction inspired oxygen_mean']
    df['Fraction inspired oxygen_max_to_mean'] = df['Fraction inspired oxygen_max'] / df['Fraction inspired oxygen_mean']
    df['Glucose_min_to_mean'] = df['Glucose_min'] / df['Glucose_mean']
    df['Glucose_max_to_mean'] = df['Glucose_max'] / df['Glucose_mean']
    df['Heart Rate_min_to_mean'] = df['Heart Rate_min'] / df['Heart Rate_mean']
    df['Heart Rate_max_to_mean'] = df['Heart Rate_max'] / df['Heart Rate_mean']
    df['Mean blood pressure_min_to_mean'] = df['Mean blood pressure_min'] / df['Mean blood pressure_mean']
    df['Mean blood pressure_max_to_mean'] = df['Mean blood pressure_max'] / df['Mean blood pressure_mean']
    df['Diastolic blood pressure_min_to_mean'] = df['Diastolic blood pressure_min'] / df['Diastolic blood pressure_mean']
    df['Diastolic blood pressure_max_to_mean'] = df['Diastolic blood pressure_max'] / df['Diastolic blood pressure_mean']
    df['Systolic blood pressure_min_to_mean'] = df['Systolic blood pressure_min'] / df['Systolic blood pressure_mean']
    df['Systolic blood pressure_max_to_mean'] = df['Systolic blood pressure_max'] / df['Systolic blood pressure_mean']
    df['Oxygen saturation_min_to_mean'] = df['Oxygen saturation_min'] / df['Oxygen saturation_mean']
    df['Oxygen saturation_max_to_mean'] = df['Oxygen saturation_max'] / df['Oxygen saturation_mean']
    df['Respiratory rate_min_to_mean'] = df['Respiratory rate_min'] / df['Respiratory rate_mean']
    df['Respiratory rate_max_to_mean'] = df['Respiratory rate_max'] / df['Respiratory rate_mean']
    df['Temperature_min_to_mean'] = df['Temperature_min'] / df['Temperature_mean']
    df['Temperature_max_to_mean'] = df['Temperature_max'] / df['Temperature_mean']
    df['Weight_min_to_mean'] = df['Weight_min'] / df['Weight_mean']
    df['Weight_max_to_mean'] = df['Weight_max'] / df['Weight_mean']
    df['pH_min_to_mean'] = df['pH_min'] / df['pH_mean']
    df['pH_max_to_mean'] = df['pH_max'] / df['pH_mean']
    return df



class FeatureTransformer(BaseEstimator, TransformerMixin):
   def __init__(self, fnames, n_neighbors=10):
      self.n_neighbors = n_neighbors
      self.fnames = fnames

   def find_outliers(self, x):
      # Outlier may contain information about the minority class, we can use it as a feature
      # We use IsolationForest to detect outlier
      knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
      x_iso_forest = knn_imputer.fit_transform(x)

      iso_forest = IsolationForest(random_state=42)
      iso_forest.fit(x_iso_forest)

      return iso_forest.predict(x_iso_forest)

   def fit(self, x, y=None):
      return self
   
   def transform(self, x, y=None):
      is_outlier = self.find_outliers(x, n_neighbors=self.n_neighbors)
      if not isinstance(x, pd.DataFrame):
         x = pd.DataFrame(x, columns=self.fnames)
      x['is_outlier'] = is_outlier
      return x
   

def cv(pipeline, x, y, scoring, k, random_state, verbose=False):
    scores = []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(x,y), start=1):

        pipe = Pipeline(pipeline)

        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        score = scoring(y_test, y_pred)
        if verbose: print(f'Fold{i}, f1 score: {score}')
        scores.append(score)

    mean = np.mean(scores)
    if verbose: print(f'Mean score: {mean}')  
    return mean


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

