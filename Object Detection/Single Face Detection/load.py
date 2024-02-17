from main import model, data, testDataSet, iou_batch, ToPILImage
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import ImageDraw 
import torch  

model.load_state_dict(torch.load('Object_Detection.pt')) 
model.eval() 

def show_bounding_box_2(image, modelOutput, originalOutput, iou): 
    _, _, o_x0, o_y0, o_x1, o_y1 = originalOutput
    _, _, m_x0, m_y0, m_x1, m_y1 = modelOutput 
    image = image.copy() 
    ImageDraw.Draw(image).rectangle(((o_x0, o_y0), (o_x1, o_y1)), outline = (0, 255, 0), width = 2) 
    ImageDraw.Draw(image).rectangle(((m_x0, m_y0), (m_x1, m_y1)), outline = (255, 0, 0), width = 2)
    plt.imshow(image)
    plt.title(f'IOU: {iou:.2f}') 
    plt.show() 

originalImage, originalLabel = testDataSet[np.random.randint(0, len(testDataSet))]  
modelOutput = model(originalImage.unsqueeze(0)) 
iou = iou_batch(modelOutput, originalLabel.unsqueeze(0)) 
_, label = ToPILImage()((originalImage, originalLabel)) 
image, modelOutput = ToPILImage()((originalImage, modelOutput.squeeze())) 
show_bounding_box_2(image, modelOutput, label, iou) 
