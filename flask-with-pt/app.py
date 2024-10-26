from flask import Flask, render_template, request
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import numpy as np

app = Flask(__name__)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model = MNISTModel()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image has a single channel (grayscale)
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

@app.route('/')
def home():
    return render_template('index.html')

@app.post('/predict')
def predict():
    if 'file' not in request.files:
        return render_template('results.html.jinja', result='No file provided')
    file = request.files['file']
    if file.filename == '':
        return render_template('results.html.jinja', result='No file provided')
    if file:
        try:
            img = Image.open(file).convert('L')
            img = transform(img).unsqueeze(0) 

            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output.data, 1)
            
            prediction = predicted.item()

            print(f'Predicted digit: {prediction}')

            return render_template('results.html.jinja', result=prediction)
        except Exception as e:
            print(f'Error during prediction: {e}')
            return render_template('results.html.jinja', result=f'Error during prediction: {e}')