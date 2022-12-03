import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

import time


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

if __name__ == "__main__":
    transforms = T.Compose([T.Resize((512,512)), T.ToPILImage(), T.ToTensor()])
    img_dataset = datasets.ImageFolder(root='./DatasetSI', transform=transforms)
    dl = DataLoader(img_dataset, batch_size=64, shuffle=True)
    dog1 = read_image('./DatasetSI/anadearmas/ana_de_armas1.jpg').resize_([3, 1024, 1024])
    dog2 = read_image(
        './DatasetSI/anadearmas/ana_de_armas2.jpg').resize_([3, 1024, 1024])
    dog_list = [dog1, dog2]
    
    
    weights = FCN_ResNet50_Weights.DEFAULT
    transforms = weights.transforms(resize_size=None)

    model = fcn_resnet50(weights=weights, progress=False)
    model = model.eval()

    batch = torch.stack([transforms(d) for d in dog_list])
    output = model(batch)['out']
    print(output.shape, output.min().item(), output.max().item())
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    dog_and_boat_masks = [
        normalized_masks[img_idx, sem_class_to_idx[cls]]
        for img_idx in range(len(dog_list))
        for cls in ('dog', 'boat')
    ]

    show(dog_and_boat_masks)

