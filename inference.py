# Imports
import pandas as pd
import numpy as np
import torch, random, gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, recall_score, balanced_accuracy_score, roc_curve

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import transformers
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW

import warnings
warnings.filterwarnings("ignore")

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

def main():
    # Load the test data.
    path = 'data/ent_task_data/evaluation.xlsx'
    df_test = pd.read_excel(path)
    df_test['label'] = df_test['label'].astype(int)

    # Parameters
    MODEL_TYPE = 'bert-base-multilingual-uncased'
    MAX_LEN = 256
    BATCH_SIZE = 128
    NUM_CORES = 4
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=True)

    # Create the dataloader
    class CompDataset(Dataset):
        def __init__(self, df):
            self.df_data = df

        def __getitem__(self, index):
            # get the sentence from the dataframe
            sentence1 = self.df_data.loc[index, 'text']
            sentence2 = self.df_data.loc[index, 'reason']
            # Process the sentence
            encoded_dict = tokenizer.encode_plus(
                        sentence1, sentence2, # Sentences to encode.
                        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                        max_length = MAX_LEN,           # Pad or truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',          # Return pytorch tensors.
                )  
            
            # These are torch tensors already.
            padded_token_list = encoded_dict['input_ids'][0]
            att_mask = encoded_dict['attention_mask'][0]
            token_type_ids = encoded_dict['token_type_ids'][0]
            
            # Convert the target to a torch tensor
            target = torch.tensor(self.df_data.loc[index, 'label'])
            target = torch.tensor(target)
            sample = (padded_token_list, att_mask, token_type_ids, target)

            return sample


        def __len__(self):
            return len(self.df_data)

    test_data = CompDataset(df_test)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                        num_workers=NUM_CORES)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_TYPE, 
        num_labels = 2, 
        output_attentions = False,
        output_hidden_states = False)

    model = model.to(device)
    print('\nTest Set...')
    model_preds_list = []
    print('Total batches:', len(test_dataloader))

    # Load the model
    path_model = f'model_{MODEL_TYPE}_{device}.bin'
    model.load_state_dict(torch.load(path_model))
    model.to(device)
    model.eval()
    # Turn off the gradient calculations.
    torch.set_grad_enabled(False)

    # Reset the total loss.
    for j, h_batch in enumerate(test_dataloader):
        inference_status = 'Batch ' + str(j + 1)
        print(inference_status, end='\r')
        b_input_ids = h_batch[0].to(device)
        b_input_mask = h_batch[1].to(device)
        b_token_type_ids = h_batch[2].to(device)     


        outputs = model(b_input_ids, 
                token_type_ids=b_token_type_ids, 
                attention_mask=b_input_mask)

        # Get the preds
        preds = outputs[0]

        # Move preds to the CPU
        val_preds = preds.detach().cpu().numpy()
        # Stack the predictions.
        if j == 0:  # first batch
            stacked_val_preds = val_preds
        else:
            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

    model_preds_list.append(stacked_val_preds)
            
    print('\nPrediction complete.')
    test_preds = np.argmax(model_preds_list[0], axis=1)

    path = 'data/ent_task_data/evaluation.xlsx'
    df_sample = pd.read_excel(path)
    df_sample['label'] = df_sample['label'].astype(int)# Assign the preds to the prediction column
    df_sample['prediction'] = test_preds

    df_sample.to_csv(f'submission_{MODEL_TYPE}.csv', index=False)
    # Check the distribution of the predicted classes.
    df_sample['prediction'].value_counts()
    score(df_sample['prediction'], df_sample['label'])

if __name__=="__main__":
    main()