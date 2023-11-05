import pandas as pd
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision
from torch import nn
import os
import cv2
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2


train_transform = A.Compose(

    [

        A.SmallestMaxSize(max_size=160),

        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),

        A.RandomCrop(height=128, width=128),

        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),

        A.RandomBrightnessContrast(p=0.5),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ]

)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

df=pd.read_csv('description.csv')
df.drop(columns=[ 'Age [year]', 'Sex',
       'Treatment technique', 'Duration [month]', 'Photo', 'Mask', 'Stereo_l',
       'Stereo_r', 'Thermal', 'Depth camera', 'Photo-thermal image',
       'Stereo map with photo-thermal image',
       'Deep map with photo-thermal image'],inplace=True)

df['Path to files']=df['Path to files'].str.replace('\\', '/')


i_transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize the image to a specific size
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
    transforms.RandomRotation(10),         # Randomly rotate the image up to 10 degrees
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define the image directory, label mapping, and the custom collate method
image_dir = ''
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

# custom_data_collator = CustomDataCollator(df, image_dir, label_mapping)

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Create custom datasets
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(image_dir, self.dataframe.iloc[idx, 0])
        #image = Image.open(f'{image_path}/photo.jpg')
        image = cv2.imread(f'{image_path}/photo.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = label_mapping[self.dataframe.iloc[idx, 1].lower()]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image.to(device), torch.tensor(label).to(device)

# Create data loaders with the custom collate method
batch_size = 64
train_loader = DataLoader(CustomDataset(train_df, transform=train_transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomDataset(test_df, transform=train_transform), batch_size=batch_size)

model=torchvision.models.convnext_base(pretrained=True)

model.classifier= nn.Sequential(
    torchvision.models.convnext.LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(in_features=1024, out_features=11, bias=True)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs=80

# Move the model to the appropriate device (e.g., GPU if available)

model.to(device)
model_save_path = "model_weights_epoch{}.pth"
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs, labels

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        torch.save(model.state_dict(), model_save_path.format(epoch))

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Validation loop
model.eval()
