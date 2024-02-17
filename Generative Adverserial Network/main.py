import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torchvision.utils import make_grid
import torch.nn.functional as F

imageSize = 64
batchSize = 60
latentSize = 100
inputColors = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [
        transforms.Resize(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

Data = datasets.MNIST('root', 'train', transform = transform, download = True)

trainDataLoader = DataLoader(Data, batch_size = batchSize, shuffle = True)

generator = nn.Sequential(
    # Layer 1
    nn.ConvTranspose2d(
        in_channels = latentSize,
        out_channels = 512, kernel_size = 4,
        stride = 1, padding = 0, bias = False
    ),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    # Layer 2
    nn.ConvTranspose2d(
        in_channels = 512,
        out_channels = 256, kernel_size = 4,
        stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    # Layer 3
    nn.ConvTranspose2d(
        in_channels = 256,
        out_channels = 128, kernel_size = 4,
        stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    # Layer 4
    nn.ConvTranspose2d(
        in_channels = 128,
        out_channels = 64, kernel_size = 4,
        stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    # Layer 5
    nn.ConvTranspose2d(
        in_channels = 64,
        out_channels = inputColors, kernel_size = 4,
        stride = 2, padding = 1, bias = False
    )
).to(device = device)

discriminator = nn.Sequential(
    nn.Conv2d(
        in_channels = inputColors, out_channels = 64,
        kernel_size = 4, stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace = True),

    # Layer 2
    nn.Conv2d(
        in_channels = 64, out_channels = 128,
        kernel_size = 4, stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace = True),

    # Layer 3
    nn.Conv2d(
        in_channels = 128, out_channels = 256,
        kernel_size = 4, stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace = True),

    # Layer 4
    nn.Conv2d(
        in_channels = 256, out_channels = 512,
        kernel_size = 4, stride = 2, padding = 1, bias = False
    ),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace = True),

    # Layer 5
    nn.Conv2d(
        in_channels = 512, out_channels = 1,
        kernel_size = 4, stride = 1, padding = 0, bias = False
    ),
    nn.Flatten(),
    nn.Sigmoid()
).to(device = device)

def trainDiscriminator(realImages, optimizer):
    optimizer.zero_grad()

    # Pass Real Data to Driscriminator
    realPreds = discriminator(realImages).view(-1)
    realLabels = torch.ones(batchSize, device = device)
    realLoss = F.binary_cross_entropy(realPreds, realLabels)
    realScore = realLoss.mean().item()

    # Generate Fake Data using Generator
    noise = torch.randn(batchSize, latentSize, 1, 1, device = device)
    generatedImage = generator(noise)

    # Pass Fake Data to Driscriminator
    fakePreds = discriminator(generatedImage).view(-1)
    fakeLabels = torch.zeros(batchSize, device = device)
    fakeLoss = F.binary_cross_entropy(fakePreds, fakeLabels)
    fakeScore = fakeLoss.mean().item()

    # update loss, weights
    loss = realLoss + fakeLoss
    loss.backward()
    optimizer.step()

    return loss.item(), realScore, fakeScore

def trainGenerator(optimizer):
    optimizer.zero_grad()

    # Generate Noise
    noise = torch.randn(batchSize, latentSize, 1, 1, device = device)
    fakeImage = generator(noise)

    # Fool the Discriminator
    preds = discriminator(fakeImage).view(-1)
    labels = torch.ones(batchSize, device = device)
    loss = F.binary_cross_entropy(preds, labels)

    # update loss and weights
    loss.backward()
    optimizer.step()

    return loss.item()

def trainModel(epochs):
    torch.cuda.empty_cache()

    D_Loss = []
    G_Loss = []
    real_Score = []
    fake_Score = []

    generator_optimizer = torch.optim.Adam(params = generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(params = discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    for epoch in range(epochs):
        for input, _ in tqdm(trainDataLoader):
            input = input.to(device = device)
            loss_d, realScore, fakeScore = trainDiscriminator(input, discriminator_optimizer)
            loss_g = trainGenerator(generator_optimizer)

        D_Loss.append(loss_d); G_Loss.append(loss_g); real_Score.append(realScore); fake_Score.append(fakeScore)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, DiscriminatorLoss: {loss_d}, GeneratorLoss: {loss_g}')
            print(f'Real Score: {realScore}, Fake Score: {fakeScore}')
            print()

    return D_Loss, G_Loss, real_Score, fake_Score

DLoss, GLoss, real_score, fake_score = trainModel(10)

noise = torch.randn(batchSize, latentSize, 1, 1, device = device)

with torch.no_grad():
    FakeImage = generator(noise).detach().cpu()

def displayExampleImages(trainSet, batchSize, title):
    plt.imshow(np.transpose(make_grid(trainSet[:batchSize], padding = 2, normalize = True), (1, 2, 0)))
    plt.axis(False); plt.title(title)
    plt.show()

displayExampleImages(FakeImage, batchSize, 'Generated Images')

originalImages = next(iter(trainDataLoader))
displayExampleImages(originalImages[0], batchSize, 'Original Images')