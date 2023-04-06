import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, recall_score, balanced_accuracy_score, roc_curve


df = pd.read_csv('combined_file.csv')

text = df['text']
reason = df['reason']
label = df['label']
length = len(text)
dfs = 3
pred = []
for i in range(length):
    preds = []
    for j in range(dfs):
        preds.append(df.iloc[i][f'pred_{j}'])
    pred.append(max(set(preds), key = preds.count))
pred = np.array(pred)
print("pred shape: ", pred.shape)
sub = {'text': text, 'reason': reason, 'label': label, 'prediction': pred}
print("Got: ", len(text), len(reason), len(label), len(pred))
sub = pd.DataFrame(sub)
print(sub.head())
sub.to_csv('ensemble.csv', index = False)

def score(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.6f}")
    print(f"Error Rate: {1-acc}")
    Recall = recall_score(y_test, y_pred, average='macro')      
    print(f"Mean Recall: {Recall}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)}")
    Array_prec_recall_f = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"Precision: {Array_prec_recall_f[0]}")
    print(f"F-Score: {Array_prec_recall_f[2]}")
          
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_test[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_test[i]!=y_pred[i]:
           FP += 1
        if y_test[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_test[i]!=y_pred[i]:
           FN += 1

    Selectivity = TN/(TN + FP)
    G_mean = np.sqrt(Selectivity*Recall) 
    print(f"Selectivity: {Selectivity}") 
    print(f"G_mean: {G_mean}")   
    
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

score(pred, label)