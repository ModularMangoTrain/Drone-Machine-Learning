import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

spatial = models.resnet50(pretrained=True)
#temporal = models.resnet50(pretrained=True)

num_classes = 2
num_epochs = 10

num_ftrs = spatial.fc.in_features
spatial.fc = nn.Linear(num_ftrs, num_classes)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spatial.to(device)


#--------------------TRAIN---------------------

train_dataset = datasets.ImageFolder(root='dataset/train', transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(spatial.parameters(), lr=0.001, momentum=0.9)

spatial.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = spatial(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

torch.save(spatial.state_dict(), "spatial_person_detector.pth")
#---------------------EVAL-----------------------
spatial.eval()

image = Image.open('test_image.jpg')
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = spatial(input_batch)
    _, predicted_idx = torch.max(output, 1)

classes = ["no_person", "person"]
print("Prediction:", classes[predicted_idx.item()])