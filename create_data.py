import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

path = 'data/ent_task_data/train.xlsx'
df_train1 = pd.read_excel(path)
df_train1['label'] = df_train1['label'].astype(int)
# df_train = df_train.sample(100)
print(df_train1.shape)

df_train1.head()

df_size = df_train1.shape[0]
df_train = df_train1

df_index = np.arange(df_size)

text_column = np.tile(df_train1["text"].values, df_size - 1)

reason_column = np.concatenate([df_train1["reason"].values[(df_index + i) % df_size] for i in range(1, df_size)])

label_column = np.zeros(df_size * (df_size - 1), dtype=int)

df_train2 = pd.DataFrame({"text": text_column, "reason": reason_column, "label": label_column})

df_train = pd.concat([df_train1, df_train2])

# for i in tqdm(range(1, df_size)):
#     for j in range(df_size):
#         df_train.append({'text': df_train1.iloc[j]['text'], 'reason': df_train1.iloc[(j+i)%df_size]['reason'], 'label': 0}, ignore_index = True)

df_train.to_csv('train.csv', index=False)
print("Done")