from main import MAIN 
import numpy as np 
import random 
import torch 
from pygame.math import Vector2 
from collections import deque 
from model import Network, DQN 

MaxMemory = 100000
learningRate = 0.01
BatchSize = 1000 

class Agent:
    def __init__(self):
        self.episilon = 0 
        self.numberOfGames = 0
        self.Memory = deque(maxlen = MaxMemory) 
        self.model = Network(11, 128, 4)  
        self.trainer = DQN(self.model, gamma = 0.4, learningRate = learningRate)  

    def getState(self, game):
        snakeHead = game.snake.body[0] 

        left_position = Vector2(snakeHead.x - 1, snakeHead.y)
        right_position = Vector2(snakeHead.x + 1, snakeHead.y) 
        up_position = Vector2(snakeHead.x, snakeHead.y - 1)
        down_position = Vector2(snakeHead.x, snakeHead.y + 1) 

        direction_left = game.snake.direction == Vector2(-1, 0) 
        direction_right = game.snake.direction == Vector2(1,0)
        direction_up = game.snake.direction == Vector2(0,-1)
        direction_down = game.snake.direction == Vector2(0,1) 

        state = [
            # Danger Straight
            (direction_left and game.checkFailAgent(right_position)) or 
            (direction_right and game.checkFailAgent(left_position)) or 
            (direction_up and game.checkFailAgent(up_position)) or 
            (direction_down and game.checkFailAgent(down_position)),

            # Danger Right 
            (direction_left and game.checkFailAgent(up_position)) or 
            (direction_right and game.checkFailAgent(down_position)) or 
            (direction_up and game.checkFailAgent(right_position)) or 
            (direction_down and game.checkFailAgent(left_position)),

            # Danger Left 
            (direction_left and game.checkFailAgent(down_position)) or 
            (direction_right and game.checkFailAgent(up_position)) or 
            (direction_up and game.checkFailAgent(left_position)) or 
            (direction_down and game.checkFailAgent(right_position)),

            # Direction the snake is moving
            direction_left, direction_right, direction_up, direction_down, 

            # Food Position 
            game.fruit.x < snakeHead.x,
            game.fruit.x > snakeHead.x,
            game.fruit.y < snakeHead.y,
            game.fruit.y > snakeHead.y 
        ]
        # print(state)

        return np.array(state, dtype = int) 

    # Get the action according to the current state
    def getAction(self, state):
        # Use random Moves upto 80 Iterations
        self.episilon = 80 - self.numberOfGames 
        finalMove = [0, 0, 0, 0] # Left | Right | Up | Down movements of Snake 
        if random.randint(0, 200) < self.episilon:
            move = random.randint(0, 3) 
            finalMove[move] = 1 
        else:
            gameState = torch.tensor(state, dtype = torch.float)
            move = self.model(gameState)
            finalMove[move.argmax().item()] = 1

        return np.array(finalMove) 
 
    def remember(self, state, action, rewards, nextState, gameOver):
        self.Memory.append((state, action, rewards, nextState, gameOver)) 

    # Data for over 1000 batches
    def longMemory(self):
        if len(self.Memory) > BatchSize:
            miniSample = random.sample(self.Memory, BatchSize) 
        else:
            miniSample = self.Memory 

        # Unzip the data 
        states, actions, rewards, nextStates, gameOver = zip(*miniSample) 
        self.trainer.train(states, actions, rewards, nextStates, gameOver) 

    # Data at each step 
    def shortMemory(self, state, action, reward, nextState, gameOver): 
        self.trainer.train(state, action, reward, nextState, gameOver)  

def trainGame():
    agent = Agent()
    game = MAIN() 

    while True:
        stateOld = agent.getState(game) 
        finalMove = agent.getAction(stateOld)
        reward, gameOver, score = game.run(finalMove) 
        newState = agent.getState(game) 
        agent.shortMemory(stateOld, finalMove, reward, newState, gameOver)
        agent.remember(stateOld, finalMove, reward, newState, gameOver)

        if gameOver:
            game.game_over() 
            agent.numberOfGames += 1 
            agent.longMemory() 

if __name__ == '__main__':
    trainGame()