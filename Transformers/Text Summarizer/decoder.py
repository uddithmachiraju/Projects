from torch import nn 
from attention import MultiHeadAttention, MultiHeadCrossAttention 
from ffn import FeedForwardNetwork

class DecoderLayer(nn.Module):
    def __init__(self, hidden_units, ffn_hidden, heads):
        super(DecoderLayer, self).__init__()
        self.MH_attention = MultiHeadAttention(hidden_units, heads) 
        self.MHC_attention = MultiHeadCrossAttention(hidden_units, heads) 
        self.layer_norm = nn.LayerNorm(hidden_units) 
        self.ffn = FeedForwardNetwork(hidden_units, ffn_hidden, heads) 
        self.dropout = nn.Dropout(0.1) 

    def forward(self, encoder_output, decoder_input, mask):
        # 1. masked Multi Head Attention Layer 
        residual = decoder_input 
        decoder_input = self.MH_attention(decoder_input, masked_input = mask) 
        decoder_input = self.dropout(decoder_input)
        decoder_input = self.layer_norm(decoder_input + residual)  

        # 2. Multi Head Attention Layer 
        residual = decoder_input 
        decoder_input = self.MHC_attention(encoder_output, decoder_input, masked_input = None) 
        decoder_input = self.dropout(decoder_input)
        decoder_input = self.layer_norm(decoder_input + residual) 

        # 3. Feed Forward Network 
        residual = decoder_input 
        decoder_input = self.ffn(decoder_input) 
        decoder_input = self.drouout(decoder_input)
        decoder_input = self.layer_norm(decoder_input + residual) 

        return decoder_input  
    
class Decoder(nn.Module):
    def __init__(self, hidden_units, ffn_hidden, heads, num_of_layers):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(*[DecoderLayer(hidden_units, ffn_hidden, heads) 
                                     for _ in range(num_of_layers)])
    def forward(self, x, y, masked_input):
        return self.layers(x, y, masked_input)   