import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torchvision.transforms.functional as tf 
from PIL import Image, ImageDraw 
from torchvision.transforms import Compose 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
from torchvision.ops import box_iou 
from tqdm import tqdm 
import os 

data = pd.read_csv('faces.csv')

def load_image_with_label(index, dataFrame = data):
    image_name = dataFrame.iloc[index]['image_name']
    image_path = str('images/' + image_name) 
    image = Image.open(image_path) 
    labels = (dataFrame.iloc[index]['width'], 
              dataFrame.iloc[index]['height'],
              dataFrame.iloc[index]['x0'],
              dataFrame.iloc[index]["y0"],
              dataFrame.iloc[index]['x1'],
              dataFrame.iloc[index]['y1']) 
    
    return image, labels 

def show_bounding_box(image, label):
    _, _, x0, y0, x1, y1 = label 
    image = image.copy()
    ImageDraw.Draw(image).rectangle(((x0, y0), (x1, y1)), outline = 'green', width = 2) 
    plt.imshow(image) 
    plt.show()

class Resize:
    def __init__(self, newSize = (256, 256)):
        self.newWidth = newSize[0]
        self.newHeight = newSize[1]

    def __call__(self, image_label_sample):
        image, label = image_label_sample 
        width, height, x0, y0, x1, y1 = label 
        newImage = tf.resize(image, (self.newWidth, self.newHeight))
        
        x0 = x0 * (self.newWidth / width)
        x1 = x1 * (self.newWidth / width)
        y0 = y0 * (self.newHeight / height)
        y1 = y1 * (self.newHeight / height) 
        width = self.newWidth
        height = self.newHeight 
        
        return newImage, (width, height, x0, y0, x1, y1)
    
class HorizontalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob 

    def __call__(self, image_label_sample):
        image, label = image_label_sample 
        width, height, x0, y0, x1, y1 = label 
        if np.random.random() < self.prob:
            image = tf.hflip(image)
            label = (width, height, width - x1, y0, width - x0, y1) 

        return image, label  
    
class VerticalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob 

    def __call__(self, image_label_sample):
        image, label = image_label_sample 
        imageWidth, imageHeight = image.size 
        width, height, x0, y0, x1, y1 = label 
        if np.random.random() < self.prob:
            image = tf.vflip(image)
            label = (width, height, x0, imageHeight - y1, x1, imageHeight - y0)
        # print("V Flip: ", label)

        return image, label  
    
class ToTensor:
    def __init__(self, scale_label = True):
        self.scale_label = scale_label 

    def __call__(self, image_label_sample):
        image, label = image_label_sample
        imageWidth, imageHeight = image.size 
        width, height, x0, y0, x1, y1 = label 
        image = tf.to_tensor(image) 
        if self.scale_label:
            width = width / imageWidth 
            height = height / imageHeight 
            newx0 = x0 / imageWidth
            newy0 = y0 / imageHeight 
            newx1 = x1 / imageWidth
            newy1 = y1 / imageHeight 
            label = width, height, newx0, newy0, newx1, newy1
            # print("Tensor :", label)
        label = torch.tensor(label, dtype = torch.float32)

        return image, label 
    
class ToPILImage:
    def __init__(self, unscale_labels = True):
        self.unscale_labels = unscale_labels 

    def __call__(self, image_label_sample):
        image, label = image_label_sample 
        image = tf.to_pil_image(image) 
        imageWidth, imageHeight = image.size 
        
        if self.unscale_labels:
            width, height, x0, y0, x1, y1 = label 

            x0 = x0 * imageWidth 
            y0 = y0 * imageHeight 
            x1 = x1 * imageWidth
            y1 = y1 * imageHeight 

            width = width * imageWidth 
            height = height * imageHeight 

            if x0 < x1: x0, x1 = x1, x0 
            if y0 < y1: y0, y1 = y1, y0 
            
            label = (width, height, x1, y1, x0, y0) 
        return image, label 
    
# image, label = load_image_with_label(np.random.randint(0, len(data))) 
# show_bounding_box(image, label) 
# transforms = Compose([Resize(), HorizontalFlip(),  VerticalFlip(), ToTensor()]) 

# newImage, newLabels = transforms((image, label)) 
# newImage, newLabels = ToPILImage()((newImage, newLabels)) 
# show_bounding_box(newImage, newLabels) 

class Data(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms 

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['image_name']
        image_path = str('images/' + image_name) 
        image = Image.open(image_path) 
        labels = (self.data.iloc[index]['width'], 
                self.data.iloc[index]['height'],
                self.data.iloc[index]['x0'],
                self.data.iloc[index]["y0"],
                self.data.iloc[index]['x1'],
                self.data.iloc[index]['y1']) 
        image, labels = self.transforms((image, labels)) 
        
        return image, labels 
    
    def __len__(self):
        return len(self.data) 
    
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__() 

        self.network_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels = in_channels,
                      kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        self.network_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) 
        )

    def forward(self, input):
        input = self.network_1(input) + input
        input = self.network_2(input) 
        return input 
    
class Network(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            ResNet(in_channels, out_channels),
            nn.MaxPool2d(2),
            ResNet(out_channels, 2 * out_channels),
            nn.MaxPool2d(2),
            ResNet(2 * out_channels, 4 * out_channels),
            nn.MaxPool2d(2),
            ResNet(4 * out_channels, 8 * out_channels),
            nn.MaxPool2d(2),
            ResNet(8 * out_channels, 16 * out_channels),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16384, 6) 
        )

    def forward(self, x):
        return self.network(x) 

def iou_batch(outputLabels, targetLabels):
    outputCordinates = outputLabels[:, 2:] 
    targetCordinates = targetLabels[:, 2:]
    return torch.trace(box_iou(outputCordinates, targetCordinates)).item() 

def batchLoss(lossFunction, output, target, optimizer = None):
    loss = lossFunction(output, target) 
    with torch.no_grad():
        iou_metric = iou_batch(output, target) 
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    return loss.item(), iou_metric 

def train_test_step(dataLoader, model, lossFunction, optimizer):
    if optimizer is not None:
        if os.path.exists('Object_Detection.pt'): model.load_state_dict(torch.load('Object_Detection.pt')); model.train() 
        else: model.train() 
    else:
        model.eval() 

    loss = 0 
    iou = 0

    for image, labels in tqdm(dataLoader): 
        outputLabels = model(image)
        lossValue, iouValue = batchLoss(lossFunction, outputLabels, labels, optimizer)
        loss += lossValue 
        iou += iouValue 

    return loss / len(dataLoader.dataset), iou / len(dataLoader.dataset)  

trainData, testData = train_test_split(data, test_size = 0.2, random_state = 42, shuffle = True)

trainTransformations = Compose([Resize(), HorizontalFlip(), VerticalFlip(), ToTensor()])
testTransformations = Compose([Resize(), ToTensor()]) 

trainDataSet = Data(trainData, trainTransformations) 
testDataSet = Data(testData, testTransformations) 

trainDataLoader = DataLoader(trainDataSet, batch_size = 8)
testDataLoader = DataLoader(testDataSet, batch_size = 16) 

model = Network(3, 16) 

lossFunction = nn.SmoothL1Loss(reduction = 'sum') 
optimizer = torch.optim.Adam(params = model.parameters(), lr = 5e-4) 

epochs = 101 
global bestLoss, loss_list, iou_list 
loss_list = {'Train': [], 'Test': []} 
iou_list = {'Train': [], 'Test': []}

def train():
    bestLoss = float('inf') 
    for epoch in range(epochs):
        trainLoss, trainIOU = train_test_step(trainDataLoader, model, lossFunction, optimizer)
        loss_list['Train'].append(trainLoss)
        iou_list['Train'].append(trainIOU) 

        with torch.inference_mode():
            testLoss, testIOU = train_test_step(testDataLoader, model, lossFunction, None)
            loss_list['Test'].append(testLoss)
            iou_list['Test'].append(testIOU) 
            if testLoss < bestLoss:
                torch.save(model.state_dict(), 'Object_Detection.pt')
                bestLoss = testLoss 

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}')
            print(f'Training Loss: {trainLoss:.6} Training IOU: {trainIOU:.6}')
            print(f'Test Loss: {testLoss:.6}, Test IOU: {testIOU:.6}') 

if __name__ == '__main__':
    train() 

    plt.plot(range(1, epochs+1), loss_list['Train'], label = 'Train')
    plt.plot(range(1, epochs + 1), loss_list['Test'],  label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 

    plt.plot(range(1, epochs + 1), iou_list['Train'], label = 'Train')
    plt.plot(range(1, epochs + 1), iou_list['Test'], label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.show() 