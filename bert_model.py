# -*- coding: utf-8 -*-

import pickle

"""## data preprocessed

Since BERT has it own tokenizer, we convert ID to real word data
"""

#load original data
def load_data(train_path='data/raw_data/atis.train.pkl',test_path='data/raw_data/atis.test.pkl'):
  #training data
  with open(train_path,'rb') as f:
    train_data = pickle.load(f)
  #test data
  with open(test_path,'rb') as f:
    test_data = pickle.load(f)
  return train_data,test_data
train_data, test_data = load_data()
#label id
train_label_ids = []
for label in train_data[0]['intent_labels']:
  label_id = label[0]
  train_label_ids.append(label_id)
test_label_ids = []
for label in test_data[0]['intent_labels']:
  label_id = label[0]
  test_label_ids.append(label_id)
#translate the id of train and test data to real word
vocab = train_data[1]['token_ids']
#ID number ---> actual word
id_word = {}
for word, id in vocab.items():
  id_word[id] = word
#we already get a dictionary with [id:word]
#now translate to sentences
train_sentence_ids = train_data[0]['query']
all_train_sentence = []
for sentence_ids in train_sentence_ids:
  #convert id to word
  words = []
  for id_num in sentence_ids:
    words.append(id_word[id_num])
  #join words to a sentence
  train_sentence = ' '.join(words)
  all_train_sentence.append(train_sentence)
#print first 3 sentences
print(f'convert {len(all_train_sentence)} training sentences')
print(f'first 3 training sentences:')
for i in range(3):
  print(all_train_sentence[i])
#test data:
test_sentence_ids = test_data[0]['query']
all_test_sentence = []
for sentence_ids in test_sentence_ids:
  words = []
  for id_num in sentence_ids:
    words.append(id_word[id_num])
  test_sentence = ' '.join(words)
  all_test_sentence.append(test_sentence)
print(f'convert {len(all_test_sentence)} test sentences')
print(f'first 3 test sentences:')
for i in range(3):
  print(all_test_sentence[i])

def create_label_mapping(labels):
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

# Create mapping for your labels
label_to_idx, idx_to_label = create_label_mapping(train_label_ids + test_label_ids)
print(f"Label mapping created. Number of classes: {len(label_to_idx)}")

# Remap your labels
train_label_ids = [label_to_idx[label] for label in train_label_ids]
test_label_ids = [label_to_idx[label] for label in test_label_ids]

"""## fine-tune BERT Model"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

all_labels = train_label_ids + test_label_ids
unique_labels = sorted(set(all_labels))
num_classes = len(unique_labels)

#initial pretrained BERT model and tokenization
token = BertTokenizer.from_pretrained('bert-base-uncased')
#get lenth of label and use on the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
#define a function tokenize the data
def tokenize_data(sentence, labels):
  #tokenized the sentence, return Pytorch tensor
  tokenized_data = token(sentence, truncation=True, padding=True, max_length=100, return_tensors='pt')
  #create tensordataset
  dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(labels))
  return dataset
#get dataset and dataloader
train_dataset = tokenize_data(all_train_sentence, train_label_ids)
test_dataset = tokenize_data(all_test_sentence, test_label_ids)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

"""Fine-tune strategy: Freezing strategy
1. Freezing all layers except classifier
2. Unfreeze last 2 layers
3. Unfreeze last 4 layers
4. Unfreeze all layers
"""

#part 1
def freeze_except_classifier(model):
  for param in model.bert.parameters():
    param.requires_grad = False
  for param in model.classifier.parameters():
    param.requires_grad = True
#part 2, 3
def unfreeze_lastnlayers(model, n):
  #pretrained BERT has 12 layers
  Bert_layer = 12
  for i in range(Bert_layer - n, Bert_layer):
    for param in model.bert.encoder.layer[i].parameters():
      param.requires_grad = True
#part 4
def unfreeze_all_layers(model):
  for param in model.bert.parameters():
    param.requires_grad = True
#define main training function
def train(model, train, test, epoch, max_lr, part, train_losses, val_losses, train_accs, val_accs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  print(f"{part} - Training on: {device}")
  #set optimizer
  optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=max_lr, weight_decay=0.01)
  #set scheduler learning rate
  steps = len(train) * epoch
  #train
  for e in range(epoch):
    model.train()
    epoch_train_loss = 0
    train_preds, train_labels = [], []
    for batch in train:
      id, attention_mask, label = batch
      id = id.to(device)
      attention_mask = attention_mask.to(device)
      label = label.to(device)
      optimizer.zero_grad()
      output = model(input_ids=id, attention_mask=attention_mask, labels=label)
      loss = output.loss
      loss.backward()
      #clip the parameter prevent explose
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      lr = max_lr * (1 - e/epoch)
      for g in optimizer.param_groups:
        g['lr'] = lr
      optimizer.step()
      epoch_train_loss += loss.item()
      train_preds.extend(torch.argmax(output.logits, dim=1).tolist())
      train_labels.extend(label.tolist())
    train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
    avg_train_loss = epoch_train_loss / len(train)
    train_accs.append(train_acc)
    train_losses.append(avg_train_loss)
    #validation
    model.eval()
    epoch_val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
      for batch in test:
        id, attention_mask, label = batch
        id = id.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        output = model(input_ids=id, attention_mask=attention_mask, labels=label)
        loss = output.loss
        epoch_val_loss += loss.item()
        val_preds.extend(torch.argmax(output.logits, dim=1).tolist())
        val_labels.extend(label.tolist())
    var_acc = np.mean(np.array(val_preds) == np.array(val_labels))
    avg_val_loss = epoch_val_loss / len(test)
    val_accs.append(var_acc)
    val_losses.append(avg_val_loss)
    print(f'{part}-epoch{e+1}:')
    print(f'train loss: {avg_train_loss}, train acc: {train_acc}')
    print(f'val loss: {avg_val_loss}, val acc: {var_acc}')
  return train_losses, val_losses, train_accs, val_accs
train_losses, val_losses, train_accs, val_accs = [],[],[],[]
#part1
#freeze all layer except classifer
freeze_except_classifier(model)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, test_loader, epoch=3, max_lr=1e-3, part='part1',
    train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs
)
#part2
#unfreeze last 2 layers
unfreeze_lastnlayers(model, 2)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, test_loader, epoch=3, max_lr=1e-4, part='part2',
    train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs
)
#part3
#unfreeze last 4 layers
unfreeze_lastnlayers(model, 4)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, test_loader, epoch=2, max_lr=1e-5, part='part3',
    train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs
)
#part4
#unfreeze all layers
unfreeze_all_layers(model)
train_losses, val_losses, train_accs, val_accs = train(
    model, train_loader, test_loader, epoch=2, max_lr=1e-5, part='part4',
    train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs
)

"""## Evaluation"""

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation (model, dataloader):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.eval()
  allpred, alllabel= [], []
  with torch.no_grad():
    for batch in dataloader:
      id, attention_mask, label = batch
      id = id.to(device)
      attention_mask = attention_mask.to(device)
      label = label.to(device)
      output = model(input_ids=id, attention_mask=attention_mask, labels=label)
      allpred.extend(torch.argmax(output.logits, dim=1).cpu().numpy())
      alllabel.extend(label.cpu().numpy())
  print(classification_report(alllabel, allpred))
  matrix = confusion_matrix(alllabel, allpred)
  plt.figure(figsize=(10, 10))
  sns.heatmap(matrix, annot=True)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()
evaluation(model,test_loader)