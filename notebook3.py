# Imports
import pandas as pd
import numpy as np
import os, torch, random, gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, recall_score, balanced_accuracy_score, roc_curve

import transformers, wandb
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    seed_everything(42)
    
    # Get Data
    path = 'data/ent_task_data/train.xlsx'
    df_train = pd.read_excel(path)
    df_train['label'] = df_train['label'].astype(int)

    # Load the test data.
    path = 'data/ent_task_data/evaluation.xlsx'
    df_test = pd.read_excel(path)
    df_test['label'] = df_test['label'].astype(int)
    # Train a BERT Model

    MODEL_TYPE = 'bert-base-multilingual-uncased'
    L_RATE = 1e-3
    MAX_LEN = 256
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    NUM_CORES = 4
    
    debug = False
    if not debug:
        wandb.init(project="enterpret", entity="gakash2001")
        wandb.config = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model": MODEL_TYPE,
        "lr": L_RATE,
        "max_len": MAX_LEN
        }
    # For GPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    print("lr: ", L_RATE)
    # Instantitate the Tokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=True)

    df_val = df_test

    # Create the dataloader
    class CompDataset(Dataset):
        def __init__(self, df):
            self.df_data = df
        def __getitem__(self, index):
            # get the sentence from the dataframe
            sentence1 = self.df_data.loc[index, 'text']
            temp = ((index+random.randint(1,2000))%2000)
            if random.randint(1,2) == 1:
                sentence2 = self.df_data.loc[index, 'reason']
                target = torch.tensor(1)
            else:
                sentence2 = self.df_data.loc[temp, 'reason']
                target = torch.tensor(0)
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
            sample = (padded_token_list, att_mask, token_type_ids, target)
            return sample

        def __len__(self):
            return len(self.df_data)
    
    class ValDataset(Dataset):
        def __init__(self, df):
            self.df_data = df
        def __getitem__(self, index):
            # get the sentence from the dataframe
            sentence1 = self.df_data.loc[index, 'text']
            sentence2 = self.df_data.loc[index, 'reason']
            target = torch.tensor(self.df_data.loc[index, 'label'])
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
            sample = (padded_token_list, att_mask, token_type_ids, target)
            return sample

        def __len__(self):
            return len(self.df_data)
    
    gc.collect()
    torch.cuda.empty_cache()
    train_data = CompDataset(df_train)
    val_data = ValDataset(df_val)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                        num_workers=NUM_CORES)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                        num_workers=NUM_CORES)
    print(len(train_dataloader))
    print(len(val_dataloader))

    padded_token_list, att_mask, token_type_ids, target = next(iter(train_dataloader))

    print(padded_token_list.shape)
    print(att_mask.shape)
    print(token_type_ids.shape)
    print(target.shape)

    padded_token_list, att_mask, token_type_ids, target = next(iter(val_dataloader))

    print(padded_token_list.shape)
    print(att_mask.shape)
    print(token_type_ids.shape)
    print(target.shape)

    # Define the model
    model = BertForSequenceClassification.from_pretrained(
        MODEL_TYPE, 
        num_labels = 2, 
        output_attentions = False,
        output_hidden_states = False)

    optimizer = AdamW(model.parameters(),
            lr = L_RATE, 
            eps = 1e-8
            )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.8)
    # Send the model to the device.
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                        num_workers=NUM_CORES)

    batch = next(iter(train_dataloader))

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_token_type_ids = batch[2].to(device)
    b_labels = batch[3].to(device)
    b_labels = b_labels.type(torch. int64)

    outputs = model(b_input_ids, 
                    token_type_ids=b_token_type_ids, 
                    attention_mask=b_input_mask,
                    labels=b_labels)

    len(outputs) # loss, pred

    preds = outputs[1].detach().cpu().numpy()

    y_true = b_labels.detach().cpu().numpy()
    y_pred = np.argmax(preds, axis=1)

    # This is the accuracy without any fine tuning.
    val_acc = accuracy_score(y_true, y_pred)

    print(val_acc)

    # The loss and preds are Torch tensors

    print(type(outputs[0]))
    print(type(outputs[1]))

    # Train the Model
    # For each epoch...
    best_val_acc = 0.0
    for epoch in range(0, NUM_EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))
        epoch_acc_scores_list = []
        stacked_val_labels = []
        targets_list = []
        # put the model into train mode
        model.train()

        # This turns gradient calculations on and off.
        torch.set_grad_enabled(True)

        # Reset the total loss for this epoch.
        total_train_loss = 0

        for i, batch in enumerate(train_dataloader):
            train_status = 'Batch ' + str(i+1) + ' of ' + str(len(train_dataloader))
            print(train_status, end='\r')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()        


            outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # Get the loss from the outputs tuple: (loss, logits)
            loss = outputs[0]

            # Calculate the total loss.
            total_train_loss = total_train_loss + loss.item()

            # Zero the gradients
            optimizer.zero_grad()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Use the optimizer to update Weights
            optimizer.step() 
            scheduler.step()
        print('Train loss:' ,total_train_loss)

        # ========================================
        #               Validation
        # ========================================

        # Put the model in evaluation mode.
        model.eval()

        # Turn off the gradient calculations.
        torch.set_grad_enabled(False)

        # Reset the total loss for this epoch.
        total_val_loss = 0

        for j, val_batch in enumerate(val_dataloader):

            val_status = 'Batch ' + str(j+1) + ' of ' + str(len(val_dataloader))

            print(val_status, end='\r')

            b_input_ids = val_batch[0].to(device)
            b_input_mask = val_batch[1].to(device)
            b_token_type_ids = val_batch[2].to(device)
            b_labels = val_batch[3].to(device)      


            outputs = model(b_input_ids, 
                    token_type_ids=b_token_type_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)

            # Get the loss from the outputs tuple: (loss, logits)
            loss = outputs[0]

            # Convert the loss from a torch tensor to a number.
            # Calculate the total loss.
            total_val_loss = total_val_loss + loss.item()

            # Get the preds
            preds = outputs[1]


            # Move preds to the CPU
            val_preds = preds.detach().cpu().numpy()

            # Move the labels to the cpu
            targets_np = b_labels.to('cpu').numpy()

            # Append the labels to a numpy list
            targets_list.extend(targets_np)

            if j == 0:  # first batch
                stacked_val_preds = val_preds
            else:
                stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

        # Calculate the validation accuracy
        y_true = targets_list
        y_pred = np.argmax(stacked_val_preds, axis=1)

        val_acc = accuracy_score(y_true, y_pred)


        epoch_acc_scores_list.append(val_acc)


        print('Val loss:' ,total_val_loss)
        print('Val acc: ', val_acc)
        score(y_pred, y_true)
        if not debug:
            wandb.log({"Train_loss": total_train_loss,"val_loss": total_val_loss, "val_acc": val_acc})
        # Save the best model
        if val_acc > best_val_acc:
            # save the model
            best_val_acc = val_acc
            model_name = f'model_{MODEL_TYPE}_{device}.bin'
            torch.save(model.state_dict(), model_name)
            print('Val acc improved. Saved model as ', model_name)

        # Use the garbage collector to save memory.
        gc.collect()

if __name__=="__main__":
    main()