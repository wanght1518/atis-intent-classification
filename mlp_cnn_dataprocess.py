# -*- coding: utf-8 -*-

import pickle

"""## Loading data, Check data structure"""

#define a function load ATIS dataset from pickle files
def load_data(train_path='data/raw_data/atis.train.pkl',test_path='data/raw_data/atis.test.pkl'):
  #training data
  with open(train_path,'rb') as f:
    train_data = pickle.load(f)
  #test data
  with open(test_path,'rb') as f:
    test_data = pickle.load(f)
  return train_data,test_data
#load data
train_data, test_data = load_data()
#get basic information
print(type(train_data))
print(len(train_data))
print(type(test_data))
print(len(test_data))
#structure check
print('---------------------------------------------')
print('Training data stucture for element 1:')
#check element 1
print(f'element 1 type:{type(train_data[0])}')
print(f'keys in element 1:{list(train_data[0].keys())}')
#check each key
for key in train_data[0]:
  print('----------------------')
  value = train_data[0][key]
  print(f'Key:{key}///Value type:{type(value)}///Value length:{len(value)}')
  #show the first time
  if len(value) > 0:
    first_item = value[0]
    print(f'first item shape:{first_item.shape}')
    print(f'first item:{(first_item)}')
#check element 2
print('---------------------------------------------')
print('Training data stucture for element 1:')
print(f'element 2 type:{type(train_data[1])}')
print(f'keys in element 2:{list(train_data[1].keys())}')
#check each key
for key in train_data[1]:
  value = train_data[1][key]
  print('----------------------')
  print(f'Key:{key}///Value type:{type(value)}///Value length:{len(value)}')
  #by checking the value type is dictionary
  #show the first key-value pairs
  print('first 5 items:')
  item_list = list(value.items())[:5]
  for i, (dkey, dvalue) in enumerate(item_list):
    print(f'{dkey}:{dvalue}')

"""## Tokenization Strategy

Given the instruction, we have three tokenization Strategy plus one strategy from L. Sun and K. A. Aksyonov, “Fine-Tuning BERT on the ATIS Dataset: Data Enhancement To
Improve Intent Classification Accuracy,” in Proc. 2024 9th International Symposium on
Computer and Information Processing Technology (ISCIPT), IEEE, 2024.
1. Word-level tokenization
2. Character-level tokenization
3. Subword tokenization
4. bert tokenization

By the given data structure, the data is already word-tokenized into IDs, we need to convert to sentences.
"""

#first of all, convert id to sentences
#training data:
#getting text from ID
#exmaple of vocab: {'word': 123, 'another': 456}
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

"""1. Word-level tokenization"""

def word_token(text):
  tokens = []
  #split with space
  for word in text.split():
    #remove BOS/EOS
    if word not in ['BOS','EOS']:
      tokens.append(word)
  return tokens
train_word_tokens = [word_token(text) for text in all_train_sentence]
test_word_tokens = [word_token(text) for text in all_test_sentence]
#check lens are equal and shape after tokeization
print(f'train_word_tokens shape:{len(train_word_tokens)}')
print(f'test_word_tokens shape:{len(test_word_tokens)}')
print(f'train_word_tokens shape:{train_word_tokens[0]}')

"""2. Character-level tokenization"""

def char_token(text):
  #remove BOS, EOS, and space
  chars = text.replace('BOS ','').replace(' EOS','').replace(' ','')
  return list(chars)
train_char_tokens = [char_token(text) for text in all_train_sentence]
test_char_tokens = [char_token(text) for text in all_test_sentence]
#check lens are equal and shape after tokeization
print(f'train_char_tokens shape:{len(train_char_tokens)}')
print(f'test_char_tokens shape:{len(test_char_tokens)}')
print(f'train_char_tokens shape:{train_char_tokens[0]}')

"""3. Heuristic suffix-based subword tokenization"""

def bpe_token(text):
  #create a list for commen subwords
  vbs = {'acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist',
      'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion',
      'ate', 'en', 'ify', 'fy', 'ize', 'ise',
      'able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical',
      'ious', 'ous', 'ish', 'ive', 'less', 'y',
      'ing', 'ed', 's', 'ly', 'est', 'age', 'hood', 'th', 'ward', 'wise'}
  tokens = []
  words = text.split()
  for word in words:
    #remove BOS and EOS
    if word not in ['BOS','EOS']:
      #set suffix is not found
      suffix = False
      #split word with suffix
      for vb in sorted(vbs, key=len, reverse=True):
        #two conditions: 1.endwith
        #2.word's len bigger than suffix's len + 1
        if word.endswith(vb) and len(word) > len(vb) + 1:
          prex = word[:-len(vb)]
          if prex: #check prex not empty
            tokens.extend([prex, vb])
            #set suffix find
            suffix = True
            break
      #if suffix not find, just return
      if not suffix:
        tokens.append(word)
  return tokens
train_bpe_tokens = [bpe_token(text) for text in all_train_sentence]
test_bpe_tokens = [bpe_token(text) for text in all_test_sentence]
#check lens are equal and shape after tokeization
print(f'train_bpe_tokens shape:{len(train_bpe_tokens)}')
print(f'test_bpe_tokens shape:{len(test_bpe_tokens)}')
print(f'train_bpe_tokens shape:{train_bpe_tokens[0]}')


"""## Vocabulary Constuction

In this session, we build a vocabulary for all unique tokens in order and map token to the ID number.
"""

from collections import Counter

#count the tokens in three tokenization
#use counter from collection to count all tokens in training data
train_word_counter = Counter()
train_char_counter = Counter()
train_bpe_counter = Counter()
for i in range(len(train_word_tokens)):
  train_word_counter.update(train_word_tokens[i])
  train_char_counter.update(train_char_tokens[i])
  train_bpe_counter.update(train_bpe_tokens[i])
print(f'{len(train_word_counter)} unique words')
print(f'{len(train_char_counter)} unique characters')
print(f'{len(train_bpe_counter)} unique subwords')
#create a frequency filter
#keep tokens appear more than two times
#we keep all the characters
freq_word, freq_bpe = [], []
for word, count in train_word_counter.items():
  if count >= 2:
    freq_word.append(word)
for token, count in train_bpe_counter.items():
  if count >= 2:
    freq_bpe.append(token)
freq_char = list(train_char_counter.keys())
print(f'freq_word:{len(freq_word)}')
print(f'freq_bpe:{len(freq_bpe)}')
print(f'freq_char:{len(freq_char)}')
#add <PAD>, <UNK>
word_vb = ['<PAD>','<UNK>'] + sorted(freq_word)
char_vb = ['<PAD>','<UNK>'] + sorted(freq_char)
bpe_vb = ['<PAD>','<UNK>'] + sorted(freq_bpe)
#create a dictionary map tokens to idnumber
word_id = {}
bpe_id = {}
char_id = {}
for i, j in enumerate(word_vb):
  word_id[j] = i
for i, j in enumerate(char_vb):
  char_id[j] = i
for i, j in enumerate(bpe_vb):
  bpe_id[j] = i

"""## Padding

Since Neural Network need fixed-size of input, but what we get is sentences of different lengths, we do padding to make the sentences have equal lenth.
"""

#first we convert the tokens to id use the dictionary
def tokentoids(tokens,diction):
  ids = []
  for token in tokens:
    if token in diction:
      ids.append(diction[token])
    else:
      ids.append(diction['<UNK>'])
  return ids
train_word_ids = [tokentoids(sentence_tokens, word_id) for sentence_tokens in train_word_tokens]
train_bpe_ids = [tokentoids(sentence_tokens, bpe_id) for sentence_tokens in train_bpe_tokens]
train_char_ids = [tokentoids(sentence_tokens, char_id) for sentence_tokens in train_char_tokens]
#we should find the max lenth then do padding
word_lenth, bpe_lenth, char_lenth = [], [], []
for id in train_word_ids:
  word_lenth.append(len(id))
for id in train_bpe_ids:
  bpe_lenth.append(len(id))
for id in train_char_ids:
  char_lenth.append(len(id))
print(max(word_lenth), max(bpe_lenth), max(char_lenth))

#define padding function
def padding(list_ids, max_len, pad_id=0):
  padded = []
  for id in list_ids:
    #check 3 condition: bigger, smaller, equal
    if len(id) > max_len:
      padded_list = id[:max_len]
    elif len(id) < max_len:
      padded_list = id + [pad_id] * (max_len - len(id))
    else:
      padded_list = id
    padded.append(padded_list)
  return padded
#use max = 46, 61, 214
train_word_padded = padding(train_word_ids, 40, word_id['<PAD>'])
train_bpe_padded = padding(train_bpe_ids, 40, bpe_id['<PAD>'])
train_char_padded = padding(train_char_ids, 214, char_id['<PAD>'])
#verify padding works
print(f'Word padded shape: ({len(train_word_padded)}, {len(train_word_padded[0])})')
print(f'BPE padded shape: ({len(train_bpe_padded)}, {len(train_bpe_padded[0])})')
print(f'Char padded shape: ({len(train_char_padded)}, {len(train_char_padded[0])})')
print(f'Word padded example: {train_word_padded[0]}')

"""Convert test data and get labels of train and test"""

#convert test data to ids use same dictronary with training data
test_word_id = [tokentoids(sentence_tokens, word_id) for sentence_tokens in test_word_tokens]
test_bpe_id = [tokentoids(sentence_tokens, bpe_id) for sentence_tokens in test_bpe_tokens]
test_char_id = [tokentoids(sentence_tokens, char_id) for sentence_tokens in test_char_tokens]
#check sentences are equal by three methods
print(f'{len(test_word_id)} = {len(test_bpe_id)} = {len(test_char_id)} sentences')
#padding the test sentences
test_word_padded = padding(test_word_id, 40, word_id['<PAD>'])
test_bpe_padded = padding(test_bpe_id, 40, bpe_id['<PAD>'])
test_char_padded = padding(test_char_id, 214, char_id['<PAD>'])
#check padded sentences size
print(f'Word padded shape: ({len(test_word_padded)}, {len(test_word_padded[0])})')
print(f'BPE padded shape: ({len(test_bpe_padded)}, {len(test_bpe_padded[0])})')
print(f'Char padded shape: ({len(test_char_padded)}, {len(test_char_padded[0])})')
print(f'Word padded example: {test_word_padded[0]}')
#extract training labels
train_label_ids = []
for label in train_data[0]['intent_labels']:
  label_id = label[0]
  train_label_ids.append(label_id)
print(f'lenth of label: {len(train_label_ids)}')
#extract testing labels
test_label_ids = []
for label in test_data[0]['intent_labels']:
  label_id = label[0]
  test_label_ids.append(label_id)
print(f'lenth of label: {len(test_label_ids)}')
#translate label to real word
def translate_label(label_ids, dictionary):
  #do the same thing before as we translate training data
  id_name = {}
  for name, id in dictionary.items():
    id_name[id] = name
  label_names = []
  for id in label_ids:
    label_names.append(id_name[id])
  return label_names
label_dict = train_data[1]['intent_ids']
train_label_names = translate_label(train_label_ids, label_dict)
test_label_names = translate_label(test_label_ids, label_dict)
#check lenth and specific label
print(f'lenth of training label: {len(train_label_names)}')
print(f'lenth of testing label: {len(test_label_names)}')
print(train_label_names[:5])

"""## Check what we get and save data

We process data in these steps:
1. translate the original data -> realword data
(the original data is already encoded, we use original dicctionary to translate)
2. use realword data to do 4 ways of tokenization in each sentences
3. we create a vocab to map the realword data to ids
4. we do padding for each sentences, make sure each sentences are same length
5. getting training test data and corresponding labels
"""

import h5py
import numpy as np

"""Since we didn't change the index of train/test data/label,
we just need to save the 2 version of data(padded data, realword data) and 2 version of label(encoded label, realword label)
"""

with h5py.File('ATIS_Word_traintest_data','w') as f:
  #convert to array before save to h5py
  f.create_dataset('x_word_train', data = np.array(train_word_padded, dtype=np.int32))
  f.create_dataset('x_word_test', data = np.array(test_word_padded, dtype=np.int32))
with h5py.File('ATIS_Char_traintest_data','w') as f:
  f.create_dataset('x_char_train', data = np.array(train_char_padded, dtype=np.int32))
  f.create_dataset('x_char_test', data = np.array(test_char_padded, dtype=np.int32))
with h5py.File('ATIS_BPE_traintest_data','w') as f:
  f.create_dataset('x_BPE_train', data = np.array(train_bpe_padded, dtype=np.int32))
  f.create_dataset('x_BPE_test', data = np.array(test_bpe_padded, dtype=np.int32))
with h5py.File('ATIS_Encoded_label','w') as f:
  f.create_dataset('y_label_train', data = np.array(train_label_ids, dtype=np.int32))
  f.create_dataset('y_label_test', data = np.array(test_label_ids, dtype=np.int32))
with h5py.File('ATIS_Realword_label','w') as f:
  real_train_label = np.array([label.encode('utf-8') for label in train_label_names], dtype='S')
  real_test_label = np.array([label.encode('utf-8') for label in test_label_names], dtype='S')
  f.create_dataset('y_real_label_train', data = real_train_label)
  f.create_dataset('y_real_label_test', data = real_test_label)