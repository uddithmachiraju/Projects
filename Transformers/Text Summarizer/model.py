import torch 
from torch import nn 
import torch.nn.functional as F 
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_units, heads, bias = False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        assert hidden_units % heads == 0, 'Hidden Units must modulo 0 to hidden units' 
    
        self.head_divisions = hidden_units // heads 
        self.number_of_heads = heads

        self.key = nn.Linear(key_size, hidden_units, bias) 
        self.query = nn.Linear(query_size, hidden_units, bias)
        self.value = nn.Linear(value_size, hidden_units, bias) 
        self.output = nn.Linear(hidden_units, hidden_units, bias)

        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        # (batch_size, max_length, hidden_units)
        key = self.key(x)  
        query = self.query(x)
        value = self.value(x) 

        # Converting Hidden units into --> heads * head_divisions 
        # (batch_size, max_length, hidden_units) --> (batch_size, max_length, heads, head_divisions) --> (batch_size, heads, max_length, head_divisions) 
        query = query.view(query.shape[0], -1, self.heads, self.head_divisions).permute(0, 2, 1, 3) 
        key = key.view(key.shape[0], -1, self.heads, self.head_divisions).permute(0, 2, 1, 3) 
        value = value.view(value.shape[0], -1, self.heads, self.head_divisions).permute(0, 2, 1, 3) 

        # Applying matmul on querys and key(Transpose) 
        # (batch_size, heads, max_length, head_divisions) matmul (batch_size, heads, head_divisions, max_length) --> (batch_size, heads, max_length, max_length)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1)) # Permuting can result in max_length * max_length

        # fill 0 with small number so it wont affect softmax 
        scores = scores.masked_fill(mask = 0, value = -1e9) 

        weights = F.softmax(scores) 
        weights = self.dropout(weights) 

        # (batch_size, head, max_length, max_length) matmul (batch_size, max_length, head, head_divisions) --> (batch_size, head, max_length, head_divisons)
        context = torch.matmul(weights, value)

        # Returning the context in its original length 
        # (batch_size, head, max_length, head_divisions) --> (batch_size, max_length, head, head_divisions) --> (batch_size, max_length, head * head_divisions)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.head_divisions) 

        # (batch_size, max_length, hidden_units) 
        return self.output(context)  

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dimentions, hidden_dimentions, output_dimentions):
        super(FeedForwardNetwork, self).__init__() 
        self.layer1 = nn.Linear(input_dimentions, hidden_dimentions)
        self.layer2 = nn.Linear(hidden_dimentions, output_dimentions) 
        self.dropout = nn.Dropout(0.1)
        self.activations_func = nn.GELU() 

    def forward(self, x):
        output = self.activations_func(self.layer1(x))
        output = self.layer2(self.dropout(output))
        return output  
    
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_units, max_length): 
        super(PositionalEncoding, self).__init__()
        position_embeds = torch.zeros(max_length, hidden_units).float() 
        position_embeds.requires_grad = False  

        for pos in range(max_length):
            for i in range(0, hidden_units, 2):
                position_embeds[pos, i] = math.sin(pos / (10000 ** ((2 * i) / hidden_units))) 
                position_embeds[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / hidden_units))) 

        self.position_embeds = position_embeds.unsqueeze(0) 

    def forward(self, x):
        return x + self.position_embeds 
    
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_units, heads, feed_forward_input, 
                 feed_forward_output):
        # Normalization layers are usually the same as the hidden units
        super(EncoderBlock, self).__init__() 
        self.layer_norm = nn.LayerNorm(hidden_units) 
        self.multi_head = MultiHeadAttention(key_size, query_size, value_size, hidden_units, heads) 
        self.ffn = FeedForwardNetwork(feed_forward_input, hidden_units, feed_forward_output)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, embeddings):
        '''Embedding --> position_encoding + embeds''' 

        interacted = self.dropout(self.multi_head(embeddings))
        # Residual Layer
        interacted = self.layer_norm(interacted + embeddings) 

        feed_forward_out = self.dropout(self.ffn(interacted))
        encoded = self.layer_norm(feed_forward_out + interacted)
        return encoded 
    
class TransformerEncoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_units, heads, feed_forward_input, 
                 feed_forward_output, max_length, layers):
        super(TransformerEncoder, self).__init__()
        self.hidden_units = hidden_units 
        self.embeds = nn.Embedding()
        self.positional_encoding = PositionalEncoding(hidden_units, max_length) 
        self.blocks = nn.Sequential()
        for layer in range(layers):
            self.blocks.add_module(f'Encoder_Block_{layer}', EncoderBlock(key_size, query_size, value_size, hidden_units, 
                                                                  heads, feed_forward_input, feed_forward_output))
    def forward(self, x):
        x = self.embeds(x) * math.sqrt(self.hidden_units) # Scaled for better convergence 
        x = self.positional_encoding(x) 
        # self.attention_weights = [None] * len(self.blocks) 
        for index, block in enumerate(self.blocks):
            x = block(x)
            # self.attention_weights[index] = block.attention.attention.attention_weights
        return x 

class DecoderLayer(nn.Module):
    def __init__(self, key_size, query_size, value_size, hidden_units, heads,
                 feed_forward_input, feed_forward_hidden, feed_forward_output, index):
        super(DecoderLayer, self).__init__() 
        self.multi_head_attention = MultiHeadAttention(key_size, query_size, value_size, hidden_units, heads)
        self.layer_norm = nn.LayerNorm(hidden_units) 
        self.ffn = FeedForwardNetwork(feed_forward_input, feed_forward_hidden, feed_forward_output) 
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        pass 
    
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_units, layers, heads):
        super(BERT, self).__init__()
        self.hidden_units = hidden_units
        self.layers = layers
        self.heads = heads 

        self.feed_forward_hidden = self.hidden_units * 4 

        self.embeddings = 0 