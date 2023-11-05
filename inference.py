import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
from torch import nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.convnext_base(pretrained=True)
model.classifier = nn.Sequential(
    torchvision.models.convnext.LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
    nn.Flatten(start_dim=1, end_dim=-1), 
    nn.Linear(in_features=1024, out_features=11, bias=True)
)
model.load_state_dict(torch.load('model_weights_epoch79.pth'))
model.eval()
model.to(device)

label_mapping = {
    'venous ulcers': 0,
    'mastectomy': 1,
    'venous insufficiency': 2, 
    'ischaemia': 3,
    'wound infection': 4,
    'amputation': 5,
    'diabetic foot': 6,
    'injury': 7,
    'lymphatic ulcers': 8,
    'lymphedema': 9,
    'arteriovenous fistula failure': 10
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize the image to a specific size
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

def predict(model, input):
    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        probabilities = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities)
        pred_label = {v: k for k, v in label_mapping.items()}[pred_class.item()]
    return pred_label


image_path = os.path.join('', "/content/patients/case_1/day_1/data/scene_1/photo.jpg") 
image = Image.open(image_path)
image = transform(image).unsqueeze(0) # transform = preprocessing
    
pred = predict(model, image)
pred