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
        self.fc1 = nn.Linear(24*(12)*(12), 51)
        
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
    
#inicia
    
transformaciones = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                        transforms.RandomRotation(
                            5, expand=False, center=None),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
train_set = ImageFolder('./new_dataset/training', transform=transformaciones)
train_loader = DataLoader(train_set, batch_size=38, shuffle=True)

test_set = ImageFolder('./new_dataset/validation', transform=transformaciones)
test_loader = DataLoader(test_set, batch_size=38, shuffle=True)

classes = ['adamsandler', 'adrianalima', 'anadearmas', 'angelinajolie', 'annehathaway', 'barackobama', 'benedictcumberbatch', 'bradpitt', 'brunomars', 'caradelevingne', 'charlesleclerc', 'chayanne', 'chrisevans', 'chrishemsworth', 'chrispine', 'chrispratt', 'chrisrock', 'christianbale', 'cristianoronaldo', 'danielricciardo', 'dannydevito', 'denzelwashington', 'dwaynejohnson', 'gigihadid', 'harrystyles', 'hughjackman', 'jackiechan', 'jamesfranco', 'jenniferconnelly', 'jenniferlawrence', 'johnnydepp', 'juliaroberts', 'katebeckinsale', 'katewinslet', 'kevinhart', 'leonardodicaprio', 'lewishamilton', 'margotrobbie', 'natalieportman', 'nicolekidman', 'queenelizabeth', 'robertdowneyjr', 'salmahayek', 'sandrabullock', 'selenagomez', 'sergioperez', 'stevecarrel', 'tobeymaguire', 'tomcruise', 'tomhanks', 'vindiesel']
gender = ['0', '1', '1', '1', '1', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '0', '0', '0', '1', '1', '1', '1', '0', '1', '1', '1', '0', '0', '0', '0', '0', '0']


def saveModel():
    path = './best_model333.pth'
    torch.save(model.state_dict(), path)

# def test_accuracy():
#     model.eval()
#     accuracy = 0.0
#     preds = []
#     total = 0.0
    
#     with torch.no_grad():
#         for data in test_loader:
#             images,labels = data
#             images = images.to(torch.device("cpu"))
#             labels = labels.to(torch.device("cpu"))
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             preds.append(predicted)
#             total += labels.size(0)
#             accuracy += (predicted == labels).sum().item()

    
#     accuracy = (100 * accuracy) / total
#     true_labels = test_set.targets
#     preds = torch.cat(preds).numpy()
#     recall = recall_score(true_labels, preds,average='micro')
#     f1 = f1_score(true_labels, preds, average='micro')
#     conf_matrix = confusion_matrix(true_labels, preds)
#     # Generate the confusion matrix
#     print(conf_matrix)
#     print("f1 score: ", f1)
#     print("recall score: ", recall)
#     return accuracy

def test_accuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    test_set = ImageFolder('./new_dataset/validation', transform=transformaciones)
    test_loader = DataLoader(test_set, batch_size=38, shuffle=True)
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images = images.to(torch.device("cpu"))
            labels = labels.to(torch.device("cpu"))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # Calculate confusion matrix
    confusion_matrix = torch.zeros((2, 2), dtype=torch.int32)
    # for i in range(len(labels)):
    #     if i < predicted.size(0):
    #         confusion_matrix[labels[i], predicted[i]] += 1
    
    # Calculate f1 score, recall, and accuracy
    f1 = f1_score(predicted, labels,average='macro')
    recall = recall_score(predicted, labels,average='macro', zero_division=0)
    precision = precision_score(predicted, labels,average='macro')
    accuracy = (100 * accuracy) / total
    print("precision score: ", precision)
    
    # Return results
    return f1, recall, accuracy

def train(num_epochs, optimizer, loss_fn):
    best_accuracy = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(torch.device("cpu"))
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (images,labels) in enumerate(train_loader, 0):
            model.train(True)
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
            
        accuracy = test_accuracy()
        print("For epoch ", epoch+1, " accuracy is: ",(accuracy))
        
        if accuracy[2] > best_accuracy:
            best_accuracy = accuracy[2]
            saveModel()
        
    
def imageshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
def testBatch():
    images, labels = next(iter(test_loader))
    imageshow(torchvision.utils.make_grid(images))
    print("Real labels: " , ' '.join('%5s' % classes[labels[j]] for j in range(38)))
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print("Predicted labels: " , ' '.join('%5s' % classes[predicted[j]] for j in range(38)))


if __name__ == '__main__':
    # print(len(classes))
    global model
    model = Network()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    train(250, optimizer, loss_fn)
    print("Finished training")
    