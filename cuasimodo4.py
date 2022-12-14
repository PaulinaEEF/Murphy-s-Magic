from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn  as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(36)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(72)
        self.conv4 = nn.Conv2d(in_channels=72, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*(64)*(64), 51)

    def forward(self, input):
        output = torch.sigmoid(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn3(self.conv3(output)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn4(self.conv4(output)))
        output = self.pool(output)
        output = output.view(-1, 128*(64)*64)
        output = self.fc1(output)
        return output
    
transformaciones = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Resize((1024, 1024)),
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                        transforms.RandomRotation(
                            5, expand=False, center=None),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
train_set = ImageFolder('./new_dataset/training', transform=transformaciones)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)

test_set = ImageFolder('./new_dataset/validation', transform=transformaciones)
test_loader = DataLoader(test_set, batch_size=10, shuffle=True)

classes = ['adamsandler', 'adrianalima', 'anadearmas', 'angelinajolie', 'annehathaway', 'barackobama', 'benedictcumberbatch', 'bradpitt', 'brunomars', 'caradelevingne', 'charlesleclerc', 'chayanne', 'chrisevans', 'chrishemsworth', 'chrispine', 'chrispratt', 'chrisrock', 'christianbale', 'cristianoronaldo', 'danielricciardo', 'dannydevito', 'denzelwashington', 'dwaynejohnson', 'gigihadid', 'harrystyles', 'hughjackman', 'jackiechan', 'jamesfranco', 'jenniferconnelly', 'jenniferlawrence', 'johnnydepp', 'juliaroberts', 'katebeckinsale', 'katewinslet', 'kevinhart', 'leonardodicaprio', 'lewishamilton', 'margotrobbie', 'natalieportman', 'nicolekidman', 'queenelizabeth', 'robertdowneyjr', 'salmahayek', 'sandrabullock', 'selenagomez', 'sergioperez', 'stevecarrel', 'tobeymaguire', 'tomcruise', 'tomhanks', 'vindiesel']
model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
device = torch.device("cpu")


def saveModel():
    path = './best_model.pth'
    torch.save(model.state_dict(), path)

def test_accuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images = images.to(torch.device("cpu"))
            labels = labels.to(torch.device("cpu"))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy) / total
    return accuracy



def train(num_epochs):
    best_accuracy = 0.0

    model.to(torch.device(device))
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (images,labels) in enumerate(train_loader, 0):
            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            if i %1000==999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                running_loss = 0.0
            
        accuracy = test_accuracy()
        print("For epoch ", epoch+1, " accuracy is: ",(accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            saveModel()


if __name__ == '__main__':
    train(50)
    print("Finished training")