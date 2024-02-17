import torch 
from torch import nn 
from funcs import scalar_dot_product 

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_units = 512, heads = 8, bias = False): 
        super(MultiHeadAttention, self).__init__() 
        '''
        hidden_units : number  of hidden units in the model
        heads : number of multi head attention heads 
        '''
        assert hidden_units % heads == 0, 'Hidden Units Modulo heads must be 0'

        self.number_of_heads = heads 
        self.hidden_units = hidden_units
        self.head_divisions = hidden_units // heads 

        self.key = nn.Linear(hidden_units, hidden_units, bias) 
        self.query = nn.Linear(hidden_units, hidden_units, bias) 
        self.value = nn.Linear(hidden_units, hidden_units, bias) 
        self.output = nn.Linear(hidden_units, hidden_units, bias)  

    def forward(self, x, masked_input = None):
        '''
        x : input for the attention layer of shape - (batch_size, max_length, hidden_units) 
        masked_input : for enoder multi head attention layer we use masked_input as None and for 
                        Decoder we use masked multi head attention layer so that our model don't see 
                        the target variables
        '''
        key = self.key(x) 
        query = self.query(x) 
        value = self.value(x) 

        # Resahpe the input from 
        # (batch_size, max_length, hidden_units) --> (batch_size, max_length, heads, head_division)
        key = key.view(key.shape[0], -1, self.number_of_heads, self.head_divisions)
        query = query.view(query.shape[0], -1, self.number_of_heads, self.head_divisions) 
        value = value.view(value.shape[0], -1, self.number_of_heads, self.head_divisions) 

        # (batch_size, max_length, heads, head_division) --> (batch_size, heads, max_length, head_divisions)
        key = torch.permute(key, (0, 2, 1, 3))
        query = torch.permute(query, (0, 2, 1, 3))
        value = torch.permute(value, (0, 2, 1, 3))

        # Scalar Dot Product between query and keys to get the scores 
        weights, values = scalar_dot_product(key, query, value, mask = masked_input) 

        # Reshaping it to (batch_size, max_length, hidden_units)
        # (batch_size, heads, max_length. hidden_units) --> (batch_size, max_length, heads, hidden_division)
        # (batch_size, max_length, hidden_units) 
        values = values.permute(dims = (0, 2, 1, 3)).contiguous().view(
            values.shape[0], -1, self.number_of_heads * self.head_divisions)

        return self.output(values) 
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_units = 512, heads = 8, bias = False):
        super(MultiHeadCrossAttention, self).__init__()
        '''
        Processing the key, value inputs to the Decoder 
        '''
        assert hidden_units % heads == 0, 'Hidden_units modulo heads must 0' 
        self.number_of_heads = heads 
        self.head_divisions = hidden_units // heads
        
        self.key = nn.Linear(hidden_units, hidden_units, bias) 
        self.query = nn.Linear(hidden_units, hidden_units, bias)
        self.values = nn.Linear(hidden_units, heads, bias) 
        self.output = nn.Linear(hidden_units, hidden_units) 

    def forward(self, x, y, masked_input):
        key = self.key(x) # key values from encoder
        value = self.values(x) # values from encoder 
        query = self.query(y) 

        key = key.view(key.shape[0], -1, self.number_of_heads, self.head_divisions)
        value = value.view(key.shape[0], -1, self.number_of_heads, self.head_divisions) 
        query = query.view(key.shape[0], -1, self.number_of_heads, self.head_divisions) 

        key = key.permute((0, 2, 1, 3))
        value = value.permute((0, 2, 1, 3))
        query = query.permute((0, 2, 1, 3)) 

        weights, values = scalar_dot_product(key, query, value, mask = masked_input) 

        values = values.permute((0, 2, 1, 3)).contiguous().view(
            values.shape[0], -1, self.number_of_heads * self.head_divisions
        )

        return self.output(values) 