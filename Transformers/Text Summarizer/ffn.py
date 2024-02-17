from torch import nn 

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_units = 512, ffn_hidden = 512 * 4):
        '''
        FeedForwardNetwork 
        Used to transfer the data in only one direction 

        hidden_units : number of hidden units in the model 
        ffn_hidden : feed forward network hidden units 
        '''
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_units, ffn_hidden)
        self.layer2 = nn.Linear(ffn_hidden, hidden_units)
        self.activation = nn.ReLU() 
        self.droput = nn.Dropout(0.1) 

    def forward(self, x):
        x = self.layer1(x) 
        x = self.activation(x) 
        x = self.droput(x)
        x = self.layer2(x) 
        return x 
    