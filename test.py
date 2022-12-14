import torch
from torchvision import transforms
import PIL.Image as Image
import torch.nn  as nn
import torch.nn.functional as F

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

def main():
    model = Network()
    model.load_state_dict(torch.load('best_model.pth'))
    classes = ['adamsandler', 'adrianalima', 'anadearmas', 'angelinajolie', 'annehathaway', 'barackobama', 'benedictcumberbatch', 'bradpitt', 'brunomars', 'caradelevingne', 'charlesleclerc', 'chayanne', 'chrisevans', 'chrishemsworth', 'chrispine', 'chrispratt', 'chrisrock', 'christianbale', 'cristianoronaldo', 'danielricciardo', 'dannydevito', 'denzelwashington', 'dwaynejohnson', 'gigihadid', 'harrystyles', 'hughjackman', 'jackiechan', 'jamesfranco', 'jenniferconnelly', 'jenniferlawrence', 'johnnydepp', 'juliaroberts', 'katebeckinsale', 'katewinslet', 'kevinhart', 'leonardodicaprio', 'lewishamilton', 'margotrobbie', 'natalieportman', 'nicolekidman', 'queenelizabeth', 'robertdowneyjr', 'salmahayek', 'sandrabullock', 'selenagomez', 'sergioperez', 'stevecarrel', 'tobeymaguire', 'tomcruise', 'tomhanks', 'vindiesel']
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

    # image_classify(model,transformaciones, './new_dataset/bk.jpg',classes)
    image_classify(model,transformaciones, './new_dataset/validation/chrispine/cp1.jpg',classes)

def image_classify(model,image_transforms, image_path, classes):
    # model in evaluation mode
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    outPut = model(image)
    _, predicted = torch.max(outPut.data,1)

    print(image_path, '\n',classes[predicted.item()])


if __name__ == "__main__":
    main()