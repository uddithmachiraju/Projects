import pandas as pd 
import numpy as np 
from torch import nn 
import re, nltk, torch 
from data import word_mapping 
import matplotlib.pyplot as plt 
from collections import Counter 
from nltk.corpus import stopwords  
from sklearn.model_selection import train_test_split  

stop_words = stopwords.words('english') 

data = pd.read_csv('Data/news_summary_more.csv', nrows = 10000)

def cleanText(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"','', text) 
    text = ' '.join([word_mapping[t] if t in word_mapping else t for t in text.split(' ')]) 
    text = re.sub(r"'s\b", "", text) 
    text = re.sub("[^a-zA-Z]", " ", text) 
    tokens = [word for word in text.split() if not word in stop_words] 

    long_words = [] 
    for word in tokens:
        if len(word) > 3: long_words.append(word) 

    new_text = ' '.join(long_words).strip() 

    def no_space(word, prev_word):
        return word in set(',!"";.''?') and prev_word != " " 
    
    new_text = new_text.replace('\u202f', ' ').replace('\xa0', ' ').lower() 
    out = [' ' + char if i > 0 and no_space(char, new_text[i - 1]) else char for i, char in enumerate(new_text)] 
    new_text = ''.join(out) 
    return new_text 

data['headlines'] = data['headlines'].apply(cleanText) 
data['text'] = data['text'].apply(cleanText) 
data['headlines'].replace('', np.nan, inplace = True) 
data['text'].replace('', np.nan, inplace = True) 
data.dropna(axis = 0, inplace = True) 

# text_lengths = [] 
# for x in data['text']:
#     text_lengths.append(len(x)) 

# summary_lengths = [] 
# for x in data['headlines']: summary_lengths.append(len(x)) 

# plt.hist(text_lengths)
# plt.hist(summary_lengths)  
# plt.show()

max_text_length = 50 
max_summary_length = 15 

xTrain, xTest, yTrain, yTest = train_test_split(data['text'], data['headlines'], test_size = 0.1, shuffle = True, random_state = 42)
# print(xTest[:1])

# Tokenize the text
def tokenize(lines, token = 'word'):
    assert token in ('word', 'char'), 'Unknown token type' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

# Add padding or remove
def padding(line, max_padding, padding_token):
    if len(line) > max_padding:
        return line[:max_padding]
    return line + [padding_token] * (max_padding - len(line)) 

class Vocab:
    def __init__(self, tokens = [], minimum_freq = 0, reserved_tokens = []):
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line] 

        counter = Counter(tokens) 

        self.token_frequency = sorted(counter.items(), key = lambda x: x[1], reverse = True)

        self.index_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, frequency in self.token_frequency if frequency >= minimum_freq])))
        
        self.token_to_index = {
            token : index 
            for index, token in enumerate(self.index_to_token)
        }

    def __len__(self):
        return len(self.index_to_token) 
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_index.get(tokens, self.unknown) 
        return [self.__getitem__(token) for token in tokens] 
    
    def to_tokens(self, indexes):
        # if multiple indexes are given 
        if hasattr(indexes, '__len__') and len(indexes) > 1:
            return [self.index_to_token[int(index)] for index in indexes]
        return self.index_to_token[indexes] 
    
    # Index for Unknown token
    def unknown(self):
        return self.token_to_index['<unk>']

source_tokens = tokenize(xTrain)
target_tokens = tokenize(yTrain)

source_vocab = Vocab(source_tokens, reserved_tokens = ['<pad>', '<bos>', '<eos>'])
target_vocab = Vocab(target_tokens, reserved_tokens = ['<pad>', '<bos>', '<eos>']) 

# function to add EOS and padding and also determine valid length 
def build_array_sum(lines, vocab, max_padding):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines] 

    array = torch.tensor([padding(l, max_padding, vocab['<pad>']) for l in lines]) 
    # valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1) 
    return array

source_array = build_array_sum(source_tokens, source_vocab, max_text_length)
target_array = build_array_sum(target_tokens, target_vocab, max_summary_length) 
data_array = (source_array, target_array)  

def load_array(data_array, batch_size, is_train = True):
    dataset = torch.utils.data.TensorDataset(*data_array)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle = is_train)

batch_size = 64 
data_iter = load_array(data_array, batch_size) 
vocab_size = len(source_vocab.token_to_index)

# feature, target = next(iter(data_iter))

feature = torch.rand((64, 8, 50, 64))
target = torch.rand((64, 8, 15, 64))

output = torch.matmul(feature, target) 
print(output.shape) 