# -*- coding: utf-8 -*-

import h5py
import numpy as np

"""## Loading preprossed data"""

#word level data
with h5py.File('ATIS_Word_traintest_data', 'r') as f:
    X_train_word = f['x_word_train'][:]
    X_test_word = f['x_word_test'][:]
#char level data
with h5py.File('ATIS_Char_traintest_data', 'r') as f:
    X_train_char = f['x_char_train'][:]
    X_test_char = f['x_char_test'][:]
#BPE level data
with h5py.File('ATIS_BPE_traintest_data', 'r') as f:
    X_train_bpe = f['x_BPE_train'][:]
    X_test_bpe = f['x_BPE_test'][:]
#encoded label
with h5py.File('ATIS_Encoded_label', 'r') as f:
    y_train_label = f['y_label_train'][:]
    y_test = f['y_label_test'][:]
#real label
with h5py.File('ATIS_Realword_label', 'r') as f:
    y_real_train = [label.decode('utf-8') for label in f['y_real_label_train'][:]]
    y_real_test = [label.decode('utf-8') for label in f['y_real_label_test'][:]]

"""## Define Evaluate function

"""

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation (model, X_test, y_test, batchsize=30):
  model.eval()
  allpred, alllabel= [], []
  with torch.no_grad():
    for i in range(0, len(X_test), batchsize):
      x_batch = X_test[i:i+batchsize]
      y_batch = y_test[i:i+batchsize]
      y_pred = model(x_batch)
      _, predicted = torch.max(y_pred.data, 1)
      allpred.extend(predicted.numpy())
      alllabel.extend(y_batch.numpy())
  preds = np.array(allpred)
  labels = np.array(alllabel)
  print(classification_report(labels, preds))
  matrix = confusion_matrix(labels, preds)
  plt.figure(figsize=(10, 10))
  sns.heatmap(matrix, annot=True)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()

"""## Implement one layer MLP with embedding

"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#split the train data to train and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_word, y_train_label, test_size=0.2, random_state=42)
#get the vocab size from token indices
vocab_size = int(max(X_train_word.max(), X_test_word.max())) + 1
print(vocab_size)

#define MLP with embedding
class MLP(nn.Module):
  def __init__(self, vocab_size, embed_dim=100, num_classes=26):
    super().__init__()
    #model architecture:
    #1. input layer
    #2. embedding layer: convert indices to dense vector
    #3. mean pooling: average all word vector in sequence
    #4. MLP classifier:
      #1)linear(100->128)
      #2)ReLU
      #3)Dropout(0.3)
      #4)Linear(128->26)
    #5. output layer
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.classifier = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
  def forward(self, x):
    emb = self.embedding(x)
    pooled = emb.mean(dim=1)
    return self.classifier(pooled)
model = MLP(vocab_size)
#shuffle data
idx = np.arange(len(X_train))
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]
#convert data to tensor
X_train_tensor = torch.LongTensor(X_train)
X_test_tensor = torch.LongTensor(X_test_word)
X_val_tensor = torch.LongTensor(X_val)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)
#set loss and optimizer
#loss function -> multi-CrossEntropyLoss
#optimizer Adam with learning rate 0.001, L2regulizer with lambda=1e-5
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#training
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epochs = 50
batch_size = 30
for epoch in range(epochs):
  model.train()
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  #shuffle batches
  batch_idx = np.random.permutation(len(X_train_tensor))
  #mini-batch training
  for i in range(0, len(X_train_tensor), batch_size):
    index = batch_idx[i:i+batch_size]
    x_batch = X_train_tensor[index]
    y_batch = y_train_tensor[index]
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * len(x_batch)
    _, predicted = torch.max(y_pred.data, 1)
    train_total += len(y_batch)
    train_correct += (predicted == y_batch).sum().item()
  train_loss = train_loss / train_total
  train_acc = train_correct / train_total
  #evaluation
  model.eval()
  val_loss = 0
  val_total = 0
  val_correct = 0
  with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
      x_batch = X_val_tensor[i:i+batch_size]
      y_batch = y_val_tensor[i:i+batch_size]
      y_pred = model(x_batch)
      loss = loss_fn(y_pred, y_batch)
      val_loss += loss.item() * len(x_batch)
      _, predicted = torch.max(y_pred.data, 1)
      val_total += len(y_batch)
      val_correct += (predicted == y_batch).sum().item()
  val_loss = val_loss / val_total
  val_acc = val_correct / val_total
  train_losses.append(train_loss)
  val_losses.append(val_loss)
  train_accs.append(train_acc)
  val_accs.append(val_acc)
  if epoch % 5 == 0 or epoch == 0:
    print(f'epoch {epoch+1}/{epochs}')
    print(f'train Loss: {train_loss:.4f}')
    print(f'train Acc: {train_acc:.4f}')
    print(f'val Loss: {val_loss:.4f}')
    print(f'val Acc: {val_acc:.4f}')
#plot the learning curve and accuracy curve
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#plot1 loss curve
ax1.plot(train_losses, 'b-', label='train Loss')
ax1.plot(val_losses, 'r-', label='val Loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('training and validation Loss')
ax1.legend()
#plot2 accuracy curve
ax2.plot(train_accs, 'b-', label='train Acc')
ax2.plot(val_accs, 'r-', label='val Acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('training and validation accuracy')
ax2.legend()
plt.tight_layout()
plt.show()
evaluation(model,X_val_tensor,y_val_tensor)

"""## Implement CNN with embedding(Word tokenization)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, vocab_size, embed_dim=100, num_classes=26, filter_num=100,
               filter_sizes=[3,4,5], dropout=0.5):
    super().__init__()
    #model architecture
    #1. input layer
    #2. embedding layer
    #3. reshape [batch,seq,embed] -> [batch,embed,seq]
    #4. CNN layers
      #1) Conv1d(100->100, k=3): capture 3 word pattern
      #2) Conv1d(100->100, k=4): capture 4 word pattern
      #3) Conv1d(100->100, k=5): capture 5 word pattern
    #5. ReLU activation + Max pooling
    #6. combine all filter output usc cat
    #7. dropout = 0.5
    #8. Linear(300 -> 26)
    #9. output layer
    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    self.conv3 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[0])
    self.conv4 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[1])
    self.conv5 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[2])
    self.convolutions = [self.conv3, self.conv4, self.conv5]
    self.dropout = nn.Dropout(dropout)
    features = filter_num * len(filter_sizes)
    self.classifier = nn.Linear(features, num_classes)
  def forward(self, x):
    embedding = self.embeddings(x)
    embedding = embedding.permute(0, 2, 1)
    c1out = F.relu(self.conv3(embedding))
    c2out = F.relu(self.conv4(embedding))
    c3out = F.relu(self.conv5(embedding))
    pooled1 = F.max_pool1d(c1out, c1out.shape[2]).squeeze(2)
    pooled2 = F.max_pool1d(c2out, c2out.shape[2]).squeeze(2)
    pooled3 = F.max_pool1d(c3out, c3out.shape[2]).squeeze(2)
    cat = torch.cat((pooled1, pooled2, pooled3), dim=1)
    return self.classifier(self.dropout(cat))
#use word tokenization
X_train, X_val, y_train, y_val = train_test_split(X_train_word, y_train_label, test_size=0.2, random_state=42)
vocab_size = int(max(X_train.max(), X_val.max())) + 1
#do train as same before
model = CNN(vocab_size)
#shuffle data
idx = np.arange(len(X_train))
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]
#convert data to tensor
X_train_tensor = torch.LongTensor(X_train)
X_test_tensor = torch.LongTensor(X_test_word)
X_val_tensor = torch.LongTensor(X_val)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)
#set loss and optimizer
#loss function -> multi-CrossEntropyLoss
#optimizer Adam with learning rate 0.001, L2regulizer with lambda=1e-5
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#training
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epochs = 30
batch_size = 30
for epoch in range(epochs):
  model.train()
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  #shuffle batches
  batch_idx = np.random.permutation(len(X_train_tensor))
  #mini-batch training
  for i in range(0, len(X_train_tensor), batch_size):
    index = batch_idx[i:i+batch_size]
    x_batch = X_train_tensor[index]
    y_batch = y_train_tensor[index]
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * len(x_batch)
    _, predicted = torch.max(y_pred.data, 1)
    train_total += len(y_batch)
    train_correct += (predicted == y_batch).sum().item()
  train_loss = train_loss / train_total
  train_acc = train_correct / train_total
  #evaluation
  model.eval()
  val_loss = 0
  val_total = 0
  val_correct = 0
  with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
      x_batch = X_val_tensor[i:i+batch_size]
      y_batch = y_val_tensor[i:i+batch_size]
      y_pred = model(x_batch)
      loss = loss_fn(y_pred, y_batch)
      val_loss += loss.item() * len(x_batch)
      _, predicted = torch.max(y_pred.data, 1)
      val_total += len(y_batch)
      val_correct += (predicted == y_batch).sum().item()
  val_loss = val_loss / val_total
  val_acc = val_correct / val_total
  train_losses.append(train_loss)
  val_losses.append(val_loss)
  train_accs.append(train_acc)
  val_accs.append(val_acc)
  if epoch % 5 == 0 or epoch == 0:
    print(f'epoch {epoch+1}/{epochs}')
    print(f'train Loss: {train_loss:.4f}')
    print(f'train Acc: {train_acc:.4f}')
    print(f'val Loss: {val_loss:.4f}')
    print(f'val Acc: {val_acc:.4f}')
#plot the learning curve and accuracy curve
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#plot1 loss curve
ax1.plot(train_losses, 'b-', label='train Loss')
ax1.plot(val_losses, 'r-', label='val Loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('training and validation Loss')
ax1.legend()
#plot2 accuracy curve
ax2.plot(train_accs, 'b-', label='train Acc')
ax2.plot(val_accs, 'r-', label='val Acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('training and validation accuracy')
ax2.legend()
plt.tight_layout()
plt.show()
evaluation(model,X_val_tensor,y_val_tensor)

"""## CNN with Embedding(character tokenization)"""

class CNN(nn.Module):
  def __init__(self, vocab_size, embed_dim=100, num_classes=26, filter_num=100,
               filter_sizes=[2,3,4], dropout=0.5):
    super().__init__()
    #model architecture
    #1. input layer
    #2. embedding layer
    #3. reshape [batch,seq,embed] -> [batch,embed,seq]
    #4. CNN layers
      #1) Conv1d(100->100, k=3): capture 3 word pattern
      #2) Conv1d(100->100, k=4): capture 4 word pattern
      #3) Conv1d(100->100, k=5): capture 5 word pattern
    #5. ReLU activation + Max pooling
    #6. combine all filter output usc cat
    #7. dropout = 0.5
    #8. Linear(300 -> 26)
    #9. output layer
    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    self.conv3 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[0])
    self.conv4 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[1])
    self.conv5 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[2])
    self.convolutions = [self.conv3, self.conv4, self.conv5]
    self.dropout = nn.Dropout(dropout)
    features = filter_num * len(filter_sizes)
    self.classifier = nn.Linear(features, num_classes)
  def forward(self, x):
    embedding = self.embeddings(x)
    embedding = embedding.permute(0, 2, 1)
    c1out = F.relu(self.conv3(embedding))
    c2out = F.relu(self.conv4(embedding))
    c3out = F.relu(self.conv5(embedding))
    pooled1 = F.max_pool1d(c1out, c1out.shape[2]).squeeze(2)
    pooled2 = F.max_pool1d(c2out, c2out.shape[2]).squeeze(2)
    pooled3 = F.max_pool1d(c3out, c3out.shape[2]).squeeze(2)
    cat = torch.cat((pooled1, pooled2, pooled3), dim=1)
    return self.classifier(self.dropout(cat))
#use char tokenization
X_train, X_val, y_train, y_val = train_test_split(X_train_char, y_train_label, test_size=0.2, random_state=42)
vocab_size = int(max(X_train.max(), X_val.max())) + 1
#do train as same before
model = CNN(vocab_size)
#shuffle data
idx = np.arange(len(X_train))
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]
#convert data to tensor
X_train_tensor = torch.LongTensor(X_train)
X_test_tensor = torch.LongTensor(X_test_word)
X_val_tensor = torch.LongTensor(X_val)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)
#set loss and optimizer
#loss function -> multi-CrossEntropyLoss
#optimizer Adam with learning rate 0.001, L2regulizer with lambda=1e-5
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#training
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epochs = 30
batch_size = 30
for epoch in range(epochs):
  model.train()
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  #shuffle batches
  batch_idx = np.random.permutation(len(X_train_tensor))
  #mini-batch training
  for i in range(0, len(X_train_tensor), batch_size):
    index = batch_idx[i:i+batch_size]
    x_batch = X_train_tensor[index]
    y_batch = y_train_tensor[index]
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * len(x_batch)
    _, predicted = torch.max(y_pred.data, 1)
    train_total += len(y_batch)
    train_correct += (predicted == y_batch).sum().item()
  train_loss = train_loss / train_total
  train_acc = train_correct / train_total
  #evaluation
  model.eval()
  val_loss = 0
  val_total = 0
  val_correct = 0
  with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
      x_batch = X_val_tensor[i:i+batch_size]
      y_batch = y_val_tensor[i:i+batch_size]
      y_pred = model(x_batch)
      loss = loss_fn(y_pred, y_batch)
      val_loss += loss.item() * len(x_batch)
      _, predicted = torch.max(y_pred.data, 1)
      val_total += len(y_batch)
      val_correct += (predicted == y_batch).sum().item()
  val_loss = val_loss / val_total
  val_acc = val_correct / val_total
  train_losses.append(train_loss)
  val_losses.append(val_loss)
  train_accs.append(train_acc)
  val_accs.append(val_acc)
  if epoch % 5 == 0 or epoch == 0:
    print(f'epoch {epoch+1}/{epochs}')
    print(f'train Loss: {train_loss:.4f}')
    print(f'train Acc: {train_acc:.4f}')
    print(f'val Loss: {val_loss:.4f}')
    print(f'val Acc: {val_acc:.4f}')
#plot the learning curve and accuracy curve
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#plot1 loss curve
ax1.plot(train_losses, 'b-', label='train Loss')
ax1.plot(val_losses, 'r-', label='val Loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('training and validation Loss')
ax1.legend()
#plot2 accuracy curve
ax2.plot(train_accs, 'b-', label='train Acc')
ax2.plot(val_accs, 'r-', label='val Acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('training and validation accuracy')
ax2.legend()
plt.tight_layout()
plt.show()
evaluation(model,X_val_tensor,y_val_tensor)

"""## CNN with Embedding(BPE tokenization)"""

class CNN(nn.Module):
  def __init__(self, vocab_size, embed_dim=100, num_classes=26, filter_num=100,
               filter_sizes=[2,3,4], dropout=0.5):
    super().__init__()
    #model architecture
    #1. input layer
    #2. embedding layer
    #3. reshape [batch,seq,embed] -> [batch,embed,seq]
    #4. CNN layers
      #1) Conv1d(100->100, k=3): capture 3 word pattern
      #2) Conv1d(100->100, k=4): capture 4 word pattern
      #3) Conv1d(100->100, k=5): capture 5 word pattern
    #5. ReLU activation + Max pooling
    #6. combine all filter output usc cat
    #7. dropout = 0.5
    #8. Linear(300 -> 26)
    #9. output layer
    self.embeddings = nn.Embedding(vocab_size, embed_dim)
    self.conv3 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[0])
    self.conv4 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[1])
    self.conv5 = nn.Conv1d(embed_dim, filter_num, kernel_size=filter_sizes[2])
    self.convolutions = [self.conv3, self.conv4, self.conv5]
    self.dropout = nn.Dropout(dropout)
    features = filter_num * len(filter_sizes)
    self.classifier = nn.Linear(features, num_classes)
  def forward(self, x):
    embedding = self.embeddings(x)
    embedding = embedding.permute(0, 2, 1)
    c1out = F.relu(self.conv3(embedding))
    c2out = F.relu(self.conv4(embedding))
    c3out = F.relu(self.conv5(embedding))
    pooled1 = F.max_pool1d(c1out, c1out.shape[2]).squeeze(2)
    pooled2 = F.max_pool1d(c2out, c2out.shape[2]).squeeze(2)
    pooled3 = F.max_pool1d(c3out, c3out.shape[2]).squeeze(2)
    cat = torch.cat((pooled1, pooled2, pooled3), dim=1)
    return self.classifier(self.dropout(cat))
#use BPE tokenization
X_train, X_val, y_train, y_val = train_test_split(X_train_bpe, y_train_label, test_size=0.2, random_state=42)
vocab_size = int(max(X_train.max(), X_val.max())) + 1
#do train as same before
model = CNN(vocab_size)
#shuffle data
idx = np.arange(len(X_train))
np.random.shuffle(idx)
X_train = X_train[idx]
y_train = y_train[idx]
#convert data to tensor
X_train_tensor = torch.LongTensor(X_train)
X_test_tensor = torch.LongTensor(X_test_word)
X_val_tensor = torch.LongTensor(X_val)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)
y_test_tensor = torch.LongTensor(y_test)
#set loss and optimizer
#loss function -> multi-CrossEntropyLoss
#optimizer Adam with learning rate 0.001, L2regulizer with lambda=1e-5
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#training
train_losses, val_losses = [], []
train_accs, val_accs = [], []
epochs = 30
batch_size = 30
for epoch in range(epochs):
  model.train()
  train_loss = 0.0
  train_correct = 0
  train_total = 0
  #shuffle batches
  batch_idx = np.random.permutation(len(X_train_tensor))
  #mini-batch training
  for i in range(0, len(X_train_tensor), batch_size):
    index = batch_idx[i:i+batch_size]
    x_batch = X_train_tensor[index]
    y_batch = y_train_tensor[index]
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * len(x_batch)
    _, predicted = torch.max(y_pred.data, 1)
    train_total += len(y_batch)
    train_correct += (predicted == y_batch).sum().item()
  train_loss = train_loss / train_total
  train_acc = train_correct / train_total
  #evaluation
  model.eval()
  val_loss = 0
  val_total = 0
  val_correct = 0
  with torch.no_grad():
    for i in range(0, len(X_val_tensor), batch_size):
      x_batch = X_val_tensor[i:i+batch_size]
      y_batch = y_val_tensor[i:i+batch_size]
      y_pred = model(x_batch)
      loss = loss_fn(y_pred, y_batch)
      val_loss += loss.item() * len(x_batch)
      _, predicted = torch.max(y_pred.data, 1)
      val_total += len(y_batch)
      val_correct += (predicted == y_batch).sum().item()
  val_loss = val_loss / val_total
  val_acc = val_correct / val_total
  train_losses.append(train_loss)
  val_losses.append(val_loss)
  train_accs.append(train_acc)
  val_accs.append(val_acc)
  if epoch % 5 == 0 or epoch == 0:
    print(f'epoch {epoch+1}/{epochs}')
    print(f'train Loss: {train_loss:.4f}')
    print(f'train Acc: {train_acc:.4f}')
    print(f'val Loss: {val_loss:.4f}')
    print(f'val Acc: {val_acc:.4f}')
#plot the learning curve and accuracy curve
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#plot1 loss curve
ax1.plot(train_losses, 'b-', label='train Loss')
ax1.plot(val_losses, 'r-', label='val Loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('training and validation Loss')
ax1.legend()
#plot2 accuracy curve
ax2.plot(train_accs, 'b-', label='train Acc')
ax2.plot(val_accs, 'r-', label='val Acc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_title('training and validation accuracy')
ax2.legend()
plt.tight_layout()
plt.show()
evaluation(model,X_val_tensor,y_val_tensor)