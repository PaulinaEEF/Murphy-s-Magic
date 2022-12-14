from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from api.models import  QueryToPredict
from api.serializers import QuerySerializer
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import get_object_or_404
import os
import json


#torch
import torch
from torchvision import transforms
import PIL.Image as Image
import torch.nn  as nn


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

def serve_image(request, image_name):
    # Get the image object from the database
    image = get_object_or_404(QueryToPredict, img=image_name)

    # Create an HTTP response with the image data
    response = HttpResponse(image.data, content_type='image/jpeg')

    # Return the response
    return response

@csrf_exempt
def image_classify(model, image_transforms, image_path, classes):
    # model in evaluation mode
    model = model.eval()

    # Open the image file and apply the preprocessing transformations
    image = Image.open(image_path)
    image = image_transforms(image).float()

    # Add an extra dimension to represent the batch size
    image = image.unsqueeze(0)

    # Use the model to make a prediction on the image
    output = model(image)

    # Get the index of the predicted class
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.item()
    
    # Return the predicted class
    prediction = {"prediction": classes[predicted]}
    return prediction

@csrf_exempt
def handlePrediction(request):
    model = Network()
    model.load_state_dict(torch.load(settings.MODEL_DIR))
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
    prediction = image_classify(model, transformaciones, request.FILES.get('img') ,classes)
    return json.dumps(prediction)


@api_view(['GET'])
def api_get_query_view(request, title):
    try:
        query_details = QueryToPredict.objects.get(title = title)
    except QueryToPredict.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        serializer = QuerySerializer(query_details)
        return Response(serializer.data)

@api_view(['POST'])
def api_create_query_view(request):
    query_details = QueryToPredict()
    try:
        if request.method == "POST":
            serializer = QuerySerializer(query_details, data= request.data)
        if serializer.is_valid():
            prediction = handlePrediction(request)
            serializer.save()
            return JsonResponse(prediction, safe=False)
        else:
            print(serializer.errors)
        
    except Exception as e:
        print(e)

@api_view(['DELETE'])
def api_delete_query_view(request, title):
    try:
        query_details = QueryToPredict.objects.get(title = title)
    except QueryToPredict.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == "DELETE":
        op = query_details.delete()
        data = {}
        if op:
            data['success'] = "Delete succesful"
        else:
            data['error'] = "Delete failed"
        return Response(data=data)