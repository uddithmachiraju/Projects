import torch 
from torch import nn 
from attention import MultiHeadAttention 
from ffn import FeedForwardNetwork 

class EncoderLayer(nn.Module):
    '''https://miro.medium.com/v2/resize:fit:1100/format:webp/1*pQDgppDUGEHZKnQZuS8-UA.png''' 
    def __init__(self, hidden_units, ffn_hidden, heads):
        super(EncoderLayer, self).__init__() 
        self.attention = MultiHeadAttention(hidden_units, heads) 
        self.layer_norm = nn.LayerNorm(hidden_units) 

        self.ff_network = FeedForwardNetwork(hidden_units, ffn_hidden)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        residual = x 
        x = self.attention(x, mask = None) 
        x = self.dropout(x)
        x = self.layer_norm(residual + x) 
        residual = x 
        x = self.ffn(x) 
        x = self.dropout(x)
        x = self.layer_norm(residual + x) 
        return x 

class Encoder(nn.Module):
    def __init__(self, hidden_units, ffn_hidden, heads, num_of_layers):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(*[EncoderLayer(hidden_units, ffn_hidden, heads) 
                                     for _ in range(num_of_layers)]) 

    def forward(self, x):
        return self.layers(x) 