import torch
# from torchvision import models, transforms
from golden_resnet import resnet50_r
from torchvision import transforms
from PIL import Image
import requests

# Load a pretrained ResNet model
model = resnet50_r("resnet50-0676ba61.pth")
print(model)
model.eval()  # Set the model to evaluation mode

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img_path = '../img/pexels-photo-210019.jpeg'
input_image = Image.open(img_path).convert("RGB")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

# Check if a GPU is available and if so, move the input and model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the top 5 categories of the image
_, indices = torch.topk(probabilities, 5)
percentage = probabilities[indices].tolist()

# Load the labels
labels = []
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
labels = response.text.split("\n")
# Print the top 5 categories
for i in range(5):
    print(labels[indices[i]], percentage[i])


# Dump the logits into a text file
with open("logits_tv_local.txt", "w") as f:
    f.write(str(output))
