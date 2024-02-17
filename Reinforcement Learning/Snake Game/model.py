import torch 
from torch import nn 
import numpy as np

class Network(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Network, self).__init__()
        self.networks = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(), 
            nn.Linear(hiddenSize, outputSize) 
        )

    def forward(self, input):
        return self.networks(input) 
  
class DQN:
    def __init__(self, model, gamma, learningRate):
        self.gamma = gamma
        self.learningRate = learningRate 
        self.model = model
        self.lossFunction = nn.MSELoss() 
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.learningRate) 

    def train(self, states, actions, rewards, nextStates, gameOver):
        states = torch.tensor(np.array(states), dtype = torch.float)
        actions = torch.tensor(np.array(actions), dtype = torch.float)
        rewards = torch.tensor(np.array(rewards), dtype = torch.long)
        nextStates = torch.tensor(np.array(nextStates), dtype = torch.float)
        # gameOver = torch.tensor(gameOver, dtype = torch.float) 

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            nextStates = torch.unsqueeze(nextStates, 0)
            gameOver = (gameOver, )
        
        preds = self.model(states)
        target = preds.clone() 

        for index in range(len(gameOver)):
            Q_new = rewards[index]
            if not gameOver[index]:
                Q_new = rewards[index] + self.gamma * torch.max(self.model(nextStates[index])) 
            target[index][torch.argmax(actions[index]).item()] = Q_new 

        self.optimizer.zero_grad()
        loss = self.lossFunction(target, preds)
        loss.backward()
        self.optimizer.step()