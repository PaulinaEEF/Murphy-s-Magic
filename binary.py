import torch
import torchvision
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*(12)*(12), 2)
        
    def forward(self, input):
        output = torch.sigmoid(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn4(self.conv4(output)))
        output = self.pool(output)
        output = torch.sigmoid(self.bn5(self.conv5(output)))
        output = self.pool(output)
        output = output.view(-1, 24*(12)*12)
        output = self.fc1(output)
        return output

# Load the data
transformaciones = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((224, 224)),
                        torchvision.transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                        torchvision.transforms.RandomRotation(
                            5, expand=False, center=None),
                        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
train_data = torchvision.datasets.ImageFolder(root='./new_dataset/binary_training',
                                              transform=transformaciones)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=38, shuffle=True)

test_data = torchvision.datasets.ImageFolder(root='./new_dataset/binary_validation',
                                             transform=transformaciones)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=38, shuffle=True)

def saveModel():
    path = './modelCheckpintBinary.pth'
    torch.save(model.state_dict(), path)

def train(num_epochs, optimizer, loss_fn):
    best_accuracy = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(torch.device("cpu"))
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        best_accuracy = 0.0
        model.train(True)
        for i, (images,labels) in enumerate(train_loader, 0):
            images = images.to(torch.device("cpu"))
            labels = labels.to(torch.device("cpu"))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            if i %1000==999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                running_loss = 0.0
            
        accuracy = evaluate(torch.device("cpu"))
        print("For epoch ", epoch+1, " accuracy is: ",(accuracy[0]), " F1: ", accuracy[3], " Recall: ", accuracy[1], " Precision: ", accuracy[2])
        
        if accuracy[0] > best_accuracy: #aqui
            best_accuracy = accuracy[0]
            saveModel()

def evaluate(device):
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

    # Calculate f1 score, recall, and accuracy
    f1 = f1_score(predicted, labels,average='macro')
    recall = recall_score(predicted, labels,average='macro', zero_division=0)
    precision = precision_score(predicted, labels,average='macro')
    accuracy = (100*accuracy) / total
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    global model
    model = Network()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    train(50, optimizer=optimizer, loss_fn=criterion)