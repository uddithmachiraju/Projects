import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter 
import re 
from sklearn.model_selection import train_test_split

data = pd.read_csv('IMDB Dataset.csv')
xValues, yValues = data['review'][:20000].values, data['sentiment'][:20000].values 

xTrain, xTest, yTrain, yTest = train_test_split(xValues, yValues, random_state = 42, test_size = 0.3) 

def PreProcess(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\s+', '', sentence)
    sentence = re.sub(r'\d', '', sentence) 
    return sentence 

def Tokenize(xTrain, yTrain, xTest, yTest):
    wordList = [] 
    for sentence in xTrain:
        for word in sentence.lower().split():
            word = PreProcess(word)
            wordList.append(word) 
    
    for sentence in xTest:
        for word in sentence.lower().split():
            word = PreProcess(word)
            wordList.append(word)

    corpus = Counter(wordList)
    sortedCorpus = sorted(corpus, key = corpus.get, reverse = True) 
    OneHotDict = {word:index for index, word in enumerate(sortedCorpus)} 

    finalTrain, finalTest = [], []

    for sentence in xTrain:
        finalTrain.append(
            [
                OneHotDict[PreProcess(word)] 
                for word in sentence.lower().split()
                if PreProcess(word) in OneHotDict 
            ]
        ) 

    for sentence in xTest:
        finalTest.append(
            [
                OneHotDict[PreProcess(word)]
                for word in sentence.lower().split()
                if PreProcess(word) in OneHotDict
            ]
        )

    encodedTrain = [1 if label == 'positive' else 0 for label in yTrain]
    encodedTest = [1 if label == 'positice' else 0 for label in yTest] 

    return finalTrain, encodedTrain, finalTest, encodedTest, OneHotDict 

def Padding(sentence, maxPadding):
    features = torch.zeros((len(sentence), maxPadding), dtype = torch.int32)
    for index, review in enumerate(sentence):
        features[index, -len(review):] = torch.tensor(review)[:maxPadding]
    return features  

def Accuracy(yPredicted, yTrue):
    correct = torch.eq(yPredicted, yTrue).sum().item()
    return correct / len(yPredicted) 

xTrain, yTrain, xTest, yTest, OneHotDict = Tokenize(xTrain, yTrain, xTest, yTest) 

xTrainPadded = Padding(xTrain, 500); xTestPadded = Padding(xTest, 500) 

TrainDataset = TensorDataset(xTrainPadded, torch.tensor(yTrain))
TestDataset = TensorDataset(xTestPadded, torch.tensor(yTest)) 

BatchSize = 5
TrainDataLoader = DataLoader(TrainDataset, shuffle = True, batch_size = BatchSize)
TestDataLoader = DataLoader(TestDataset, shuffle = True, batch_size = BatchSize) 

class SentimentAnalysis(nn.Module):
    def __init__(self, vocabSize, embeddingDimentions, hiddenDimentions, outputSize, numberOfLayers):
        super(SentimentAnalysis, self).__init__()
        self.hiddenDimentions = hiddenDimentions
        self.numberOfLayers = numberOfLayers

        self.Embedding = nn.Embedding(vocabSize, embeddingDimentions) 
        self.LSTM = nn.LSTM(     
            input_size = embeddingDimentions, 
            hidden_size = hiddenDimentions,
            num_layers = numberOfLayers,
            batch_first = True
        )

        self.layer1 = nn.Linear(hiddenDimentions, 128)
        self.layer2 = nn.Linear(128, outputSize) 
        self.Dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, input, hidden):
        embeds = self.Embedding(input)
        lstm_out, hidden = self.LSTM(embeds, hidden)
        output = self.Dropout(lstm_out)
        output = self.layer1(output)
        output = self.Dropout(output)
        output = self.layer2(output)
        output = self.sigmoid(output)
        output = output.view(BatchSize, -1) 
        output = output[:,-1]
        return output, hidden 

    def initHidden(self, batchSize):
        h0 = torch.zeros((self.numberOfLayers, batchSize, self.hiddenDimentions))
        c0 = torch.zeros((self.numberOfLayers, batchSize, self.hiddenDimentions))
        return (h0, c0) 

vocabSize = len(OneHotDict); embeddingDimentions = 64; hiddenDimentions = 128
outputSize = 2; numberOfLayers = 4 

model = SentimentAnalysis(
    vocabSize, embeddingDimentions, hiddenDimentions, outputSize, numberOfLayers
)

lossFunction = nn.BCELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 3e-4)

def TrainAndEvaluateModel(model, optimizer, 
                          lossFunction, trainDataLoader, 
                          testDataLoader, epochs):
    for epoch in tqdm(range(epochs)):
        model.train()

        trainAccuracy = 0.0
        trainLosses = [] 
        hidden = model.initHidden(BatchSize)

        for input, label in trainDataLoader:
            h = tuple([each.data for each in hidden]) 
            output, h = model(input, hidden)
            loss = lossFunction(output.squeeze(), label.float())
            loss.backward()
            accuracy = Accuracy(torch.round(output), label)
            optimizer.step() 
        
            trainLosses.append(loss.item()) 
            trainAccuracy += accuracy 
        
        TrainLoss = np.mean(trainLosses)
        TrainAccuracy = trainAccuracy / len(trainDataLoader) 

        hidden = model.initHidden(BatchSize)
        testAccuracy = 0.0
        testLosses = [] 

        for input, label in testDataLoader:
            h = tuple([each.data for each in hidden]) 
            output, h = model(input, h)
            loss = lossFunction(output.squeeze(), label.float())
            accuracy = Accuracy(torch.round(output), label)

            testLosses.append(loss.item())
            testAccuracy += accuracy

        TestLoss = np.mean(testLosses)
        TestAccuracy = testAccuracy / len(testDataLoader) 

        print(f'Epoch: {epoch}')
        print(f'Train Loss: {TrainLoss}, Train Accuracy: {TrainAccuracy:2f}') 
        print(f'Test Loss: {TestLoss} Test Accuracy: {TestAccuracy:2f}') 

TrainAndEvaluateModel(model, optimizer, lossFunction, 
                      TrainDataLoader, TestDataLoader, 5) 