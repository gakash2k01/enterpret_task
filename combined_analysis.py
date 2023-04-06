import numpy as np
import pandas as pd

notebooks = [
    'submission_bert-large-uncased.csv',
    'submission_bert-base-uncased.csv',
    'submission_bert-base-multilingual-uncased.csv'
]
dfs = []
preds = []
for i in range(len(notebooks)):
    df_temp = pd.read_csv(notebooks[i])
    dfs.append(df_temp)
    preds.append([])
text = []
reason = []
given_label = dfs[0]['label']
label = []
length = len(given_label)
for i in range(length):
    print('Working', i, '/', length, end = '\r')
    for df in dfs:
        if True:
            for j in range(len(dfs)):
                preds[j].append(dfs[j].iloc[i]['prediction'])
            text.append(df.iloc[i]['text'])
            reason.append(df.iloc[i]['reason'])
            label.append(df.iloc[i]['label'])
            break
preds = np.array(preds)
print("pred shape: ", preds.shape)
sub = {'text': text, 'reason': reason, 'label': label}
for i in range(preds.shape[0]):
    sub[f'pred_{i}'] = preds[i]
print("Got: ", len(text), len(reason), len(label), len(sub['pred_0']))
sub = pd.DataFrame(sub)

cnt = 0
for index, row in sub.iterrows():
  if not (row['pred_0'] == row['pred_1'] and row['pred_1'] == row['pred_2']):
    cnt+=1
print("Number of different prediction:", cnt)

sub.to_csv('combined_file.csv', index = False)