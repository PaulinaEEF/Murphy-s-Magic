import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
import random

if __name__ == "__main__":
    transforms = T.Compose([T.Resize((512,512)), T.ToTensor()])
    data = datasets.ImageFolder(root='./DatasetSI', transform=transforms)
    data_loader = DataLoader(data, batch_size=64, shuffle=True)
    
    train_features, train_labels = next(iter(data_loader))
    img = train_features[0].permute(1, 2, 0)
    label = train_labels[0]
    print("Label: ", label)
    plt.imshow(img)
    plt.show()