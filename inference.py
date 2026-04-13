import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json
import base64


class CNN_Model(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def model_fn(model_dir):
    models = {}

    for crop in ["Banana", "Corn", "Grapes"]:
        model = CNN_Model(4)
        model.load_state_dict(torch.load(f"{model_dir}/{crop}_disease.pth", map_location="cpu"))
        model.eval()
        models[crop] = model

    return models


transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


#  INPUT

def input_fn(request_body, request_content_type):

    if request_content_type == 'application/json':

        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")

        data = json.loads(request_body)

        crop = data.get("crop")

        image_base64 = data.get("image")
        image_bytes = base64.b64decode(image_base64)

        return image_bytes, crop

    raise ValueError(f"Unsupported content type: {request_content_type}")


#  PREDICT

def predict_fn(input_data, models):
    image_bytes, crop = input_data

    if crop not in models:
        raise ValueError(f"Model for crop '{crop}' not found.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    model = models[crop]

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()

    labels = {
        "Banana": ["cordana", "healthy", "pestalotiopsis", "sigatoka"],
        "Corn": ["Common_Rust", "Gray_Leaf_Spot", "Healthy", "Northern_Leaf_Blight"],
        "Grapes": ["Black_Root", "Esca", "Leaf_Blight", "Healthy"]
    }

    return labels[crop][pred], crop


#  OUTPUT

def output_fn(prediction_data, content_type):
    prediction, crop_name = prediction_data

    return json.dumps({
        "prediction": prediction,
        "crop": crop_name
    })