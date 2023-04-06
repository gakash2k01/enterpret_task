# Imports

import pandas as pd
import numpy as np
import os, torch, random, gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# set a seed value
torch.manual_seed(42)
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import transformers, wandb
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW

import warnings
warnings.filterwarnings("ignore")



def main():
    # wandb.init(project="enterpret", entity="gakash2001")
    
    # Get Data
    # Load the training data.
    path = 'data/ent_task_data/train.xlsx'
    df_train = pd.read_excel(path)
    df_train['label'] = df_train['label'].astype(int)
    # df_train = df_train.sample(100)
    print(df_train.shape)
    
    df_train.head()

    # Load the test data.

    path = 'data/ent_task_data/evaluation.xlsx'
    df_test = pd.read_excel(path)
    df_test['label'] = df_test['label'].astype(int)
    print(df_test.shape)

    df_test.head()

    # Train a BERT Model

    MODEL_TYPE = 'xlm-roberta-large'

    L_RATE = 1e-6
    MAX_LEN = 256
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    NUM_CORES = 4

    # wandb.config = {
    # "epochs": NUM_EPOCHS,
    # "batch_size": BATCH_SIZE,
    # "model": MODEL_TYPE,
    # "lr": L_RATE,
    # "max_len": MAX_LEN
    # }
    # For GPU
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("DEVICE:", device)

    # Instantitate the Tokenizer

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    MODEL_TYPE1 = 'xlm-roberta-large'
    df_val = df_test
    # df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_df_len = df_train.shape[0]
    # Create the dataloader
    class CompDataset(Dataset):

        def __init__(self, df):
            self.df_data = df



        def __getitem__(self, index):
            # get the sentence from the dataframe
            sentence1 = self.df_data.loc[index, 'text']
            temp = (index+random.randint(1,2000))%1000
            if random.randint(1,2) == 1:
                sentence2 = self.df_data.loc[index, 'reason']
                target = 1
            else:
                sentence2 = self.df_data.loc[temp, 'reason']
                target = 0
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
            token_type_ids = att_mask
            
            # Convert the target to a torch tensor
            target = torch.tensor(target)
            sample = (padded_token_list, att_mask, token_type_ids, target)

            return sample


        def __len__(self):
            return len(self.df_data)
        # Test the testloader

    train_data = CompDataset(df_train)
    val_data = CompDataset(df_val)
    test_data = CompDataset(df_test)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                        num_workers=NUM_CORES)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                        num_workers=NUM_CORES)

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                        num_workers=NUM_CORES)



    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))

    # Get one train batch

    padded_token_list, att_mask, token_type_ids, target = next(iter(train_dataloader))

    print(padded_token_list.shape)
    print(att_mask.shape)
    print(token_type_ids.shape)
    print(target.shape)

    # %% [code] {"papermill":{"duration":0.219324,"end_time":"2023-02-06T13:13:29.004986","exception":false,"start_time":"2023-02-06T13:13:28.785662","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.052930Z","iopub.status.idle":"2023-02-06T15:20:58.053886Z","shell.execute_reply.started":"2023-02-06T15:20:58.053631Z","shell.execute_reply":"2023-02-06T15:20:58.053657Z"},"jupyter":{"outputs_hidden":false}}
    # Get one val batch

    padded_token_list, att_mask, token_type_ids, target = next(iter(val_dataloader))

    print(padded_token_list.shape)
    print(att_mask.shape)
    print(token_type_ids.shape)
    print(target.shape)

    # %% [code] {"papermill":{"duration":0.237602,"end_time":"2023-02-06T13:13:29.252037","exception":false,"start_time":"2023-02-06T13:13:29.014435","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.055543Z","iopub.status.idle":"2023-02-06T15:20:58.056024Z","shell.execute_reply.started":"2023-02-06T15:20:58.055769Z","shell.execute_reply":"2023-02-06T15:20:58.055794Z"},"jupyter":{"outputs_hidden":false}}
    # Get one test batch

    padded_token_list, att_mask, token_type_ids, target = next(iter(test_dataloader))

    print(padded_token_list.shape)
    print(att_mask.shape)
    print(token_type_ids.shape)
    print(target.shape)

    # %% [markdown] {"papermill":{"duration":0.008675,"end_time":"2023-02-06T13:13:29.270327","exception":false,"start_time":"2023-02-06T13:13:29.261652","status":"completed"},"tags":[]}
    # # Define the model

    # %% [code] {"papermill":{"duration":36.179148,"end_time":"2023-02-06T13:14:05.458585","exception":false,"start_time":"2023-02-06T13:13:29.279437","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.057848Z","iopub.status.idle":"2023-02-06T15:20:58.058326Z","shell.execute_reply.started":"2023-02-06T15:20:58.058074Z","shell.execute_reply":"2023-02-06T15:20:58.058098Z"},"jupyter":{"outputs_hidden":false}}
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_TYPE, 
        num_labels = 2, 
        output_attentions = False,
        output_hidden_states = False)
    # print("Token size: ", tokenizer.vocab_size, "Model: ", model.vocab.size)
    # model.vocab_size = tokenizer.vocab_size
    # %% [code] {"papermill":{"duration":0.024232,"end_time":"2023-02-06T13:14:05.495673","exception":false,"start_time":"2023-02-06T13:14:05.471441","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.060216Z","iopub.status.idle":"2023-02-06T15:20:58.060718Z","shell.execute_reply.started":"2023-02-06T15:20:58.060468Z","shell.execute_reply":"2023-02-06T15:20:58.060492Z"},"jupyter":{"outputs_hidden":false}}
    optimizer = AdamW(model.parameters(),
            lr = L_RATE, 
            eps = 1e-8
            )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)

    # Send the model to the device.
    model = model.to(device)

    # # Test the Model

    # Get one train batch

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

    # # Inspect the model output

    outputs

    len(outputs) # loss, pred

    # %% [code] {"papermill":{"duration":0.019768,"end_time":"2023-02-06T13:14:12.069581","exception":false,"start_time":"2023-02-06T13:14:12.049813","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.076741Z","iopub.status.idle":"2023-02-06T15:20:58.077591Z","shell.execute_reply.started":"2023-02-06T15:20:58.077316Z","shell.execute_reply":"2023-02-06T15:20:58.077340Z"},"jupyter":{"outputs_hidden":false}}
    preds = outputs[1].detach().cpu().numpy()

    y_true = b_labels.detach().cpu().numpy()
    y_pred = np.argmax(preds, axis=1)

    y_pred

    # %% [code] {"papermill":{"duration":0.019094,"end_time":"2023-02-06T13:14:12.098097","exception":false,"start_time":"2023-02-06T13:14:12.079003","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.078990Z","iopub.status.idle":"2023-02-06T15:20:58.079841Z","shell.execute_reply.started":"2023-02-06T15:20:58.079588Z","shell.execute_reply":"2023-02-06T15:20:58.079613Z"},"jupyter":{"outputs_hidden":false}}
    # This is the accuracy without any fine tuning.

    val_acc = accuracy_score(y_true, y_pred)

    print(val_acc)

    # %% [code] {"papermill":{"duration":0.018296,"end_time":"2023-02-06T13:14:12.125810","exception":false,"start_time":"2023-02-06T13:14:12.107514","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.081228Z","iopub.status.idle":"2023-02-06T15:20:58.082070Z","shell.execute_reply.started":"2023-02-06T15:20:58.081809Z","shell.execute_reply":"2023-02-06T15:20:58.081834Z"},"jupyter":{"outputs_hidden":false}}
    # The loss and preds are Torch tensors

    print(type(outputs[0]))
    print(type(outputs[1]))

    # %% [markdown] {"papermill":{"duration":0.009294,"end_time":"2023-02-06T13:14:12.144571","exception":false,"start_time":"2023-02-06T13:14:12.135277","status":"completed"},"tags":[]}
    # # Train the Model

    # %% [code] {"papermill":{"duration":0.017331,"end_time":"2023-02-06T13:14:12.171515","exception":false,"start_time":"2023-02-06T13:14:12.154184","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.083450Z","iopub.status.idle":"2023-02-06T15:20:58.084279Z","shell.execute_reply.started":"2023-02-06T15:20:58.084025Z","shell.execute_reply":"2023-02-06T15:20:58.084051Z"},"jupyter":{"outputs_hidden":false}}
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # %% [code] {"papermill":{"duration":789.23475,"end_time":"2023-02-06T13:27:21.415673","exception":false,"start_time":"2023-02-06T13:14:12.180923","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.085799Z","iopub.status.idle":"2023-02-06T15:20:58.086468Z","shell.execute_reply.started":"2023-02-06T15:20:58.086197Z","shell.execute_reply":"2023-02-06T15:20:58.086222Z"},"jupyter":{"outputs_hidden":false}}
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

            # Convert the loss from a torch tensor to a number.
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
        # wandb.log({"Train_loss": total_train_loss,"val_loss": total_val_loss, "al_acc": val_acc})
        # Save the best model
        if val_acc > best_val_acc:
            # save the model
            best_val_acc = val_acc
            model_name = f'model{MODEL_TYPE1}.bin'
            torch.save(model.state_dict(), model_name)
            print('Val acc improved. Saved model as ', model_name)

        # Use the garbage collector to save memory.
        gc.collect()

    # # %% [code] {"papermill":{"duration":39.776933,"end_time":"2023-02-06T13:28:01.735222","exception":false,"start_time":"2023-02-06T13:27:21.958289","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.087923Z","iopub.status.idle":"2023-02-06T15:20:58.089030Z","shell.execute_reply.started":"2023-02-06T15:20:58.088766Z","shell.execute_reply":"2023-02-06T15:20:58.088792Z"},"jupyter":{"outputs_hidden":false}}
    # # ========================================
    # #               Test Set
    # # ========================================

    # print('\nTest Set...')
    # model_preds_list = []
    # print('Total batches:', len(test_dataloader))

    # # Load the model
    # path_model = 'model.bin'
    # # model.load_state_dict(torch.load(path_model))

    # # Send the model to the device.
    # model.to(device)

    # stacked_val_labels = []


    # # Put the model in evaluation mode.
    # model.eval()

    # # Turn off the gradient calculations.
    # torch.set_grad_enabled(False)

    # # Reset the total loss.
    # total_val_loss = 0

    # for j, h_batch in enumerate(test_dataloader):

    #     inference_status = 'Batch ' + str(j + 1)

    #     print(inference_status, end='\r')

    #     b_input_ids = h_batch[0].to(device)
    #     b_input_mask = h_batch[1].to(device)
    #     b_token_type_ids = h_batch[2].to(device)     


    #     outputs = model(b_input_ids, 
    #             token_type_ids=b_token_type_ids, 
    #             attention_mask=b_input_mask)

    #     # Get the preds
    #     preds = outputs[0]


    #     # Move preds to the CPU
    #     val_preds = preds.detach().cpu().numpy()


    #     # Stack the predictions.

    #     if j == 0:  # first batch
    #         stacked_val_preds = val_preds

    #     else:
    #         stacked_val_preds = np.vstack((stacked_val_preds, val_preds))


    # model_preds_list.append(stacked_val_preds)
        
                
    # print('\nPrediction complete.')
    # test_preds = np.argmax(model_preds_list[0], axis=1)

    # # %% [code] {"papermill":{"duration":0.218691,"end_time":"2023-02-06T13:28:02.551954","exception":false,"start_time":"2023-02-06T13:28:02.333263","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.095398Z","iopub.status.idle":"2023-02-06T15:20:58.096477Z","shell.execute_reply.started":"2023-02-06T15:20:58.096212Z","shell.execute_reply":"2023-02-06T15:20:58.096237Z"},"jupyter":{"outputs_hidden":false}}
    # # Load the sample submission.
    # # The row order in the test set and the sample submission is the same.

    # path = 'data/ent_task_data/evaluation.xlsx'
    # df_sample = pd.read_excel(path)
    # df_sample['label'] = df_sample['label'].astype(int)# Assign the preds to the prediction column
    # df_sample['prediction'] = test_preds

    # df_sample.head()

    # # %% [code] {"papermill":{"duration":0.195193,"end_time":"2023-02-06T13:28:03.295712","exception":false,"start_time":"2023-02-06T13:28:03.100519","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2023-02-06T15:20:58.101626Z","iopub.execute_input":"2023-02-06T15:20:58.102512Z","iopub.status.idle":"2023-02-06T15:20:58.155858Z","shell.execute_reply.started":"2023-02-06T15:20:58.102473Z","shell.execute_reply":"2023-02-06T15:20:58.154918Z"},"jupyter":{"outputs_hidden":false}}
    # # Create a submission csv file
    # df_sample.to_csv('submission.csv', index=False)
    # # Check the distribution of the predicted classes.
    # df_sample['prediction'].value_counts()

    # val_acc = accuracy_score(df_sample['label'], df_sample['prediction'])
    # print("Test accuracy: ",val_acc)

if __name__=="__main__":
    main()