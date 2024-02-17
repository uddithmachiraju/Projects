from torch import nn 
from encoder import Encoder
from decoder import Decoder 
from os import path 

class Transformer(nn.Module):
    def __init__(self, hidden_units, ffn_hidden, heads, number_of_layers, vocab_size):
        super(Transformer, self).__init__() 
        self.encoder = Encoder(hidden_units, ffn_hidden, heads, number_of_layers)
        self.decoder = Decoder(hidden_units, ffn_hidden, heads, number_of_layers)
        self.output = nn.Linear(hidden_units, vocab_size)

    def forward(self, x, y, masked_input):
        encoder_outputs = self.encoder(x) 
        decoder_outputs = self.decoder(encoder_outputs, y, masked_input) 
        return self.output(decoder_outputs) 
