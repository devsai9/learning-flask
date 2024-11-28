from flask import Flask, render_template, request
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import os

app = Flask(__name__)

if not os.path.exists('static'): 
    os.makedirs('static')

# Model 1
class MNISTModel1(nn.Module):
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
    
model1 = MNISTModel1()
model1.load_state_dict(torch.load('model1.pth', weights_only=True))
model1.eval()

# Model 2
class MNISTModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model2 = MNISTModel2()
model2.load_state_dict(torch.load('model2.pth', weights_only=True))
model2.eval()

# Model 3
class MNISTModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model3 = MNISTModel3()
model3.load_state_dict(torch.load('model3.pth', weights_only=True))
model3.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.post('/predict')
def predict():
    if 'file' not in request.files: return render_template('results.html.jinja', result='No file provided')

    file = request.files['file']
    if file.filename == '': return render_template('results.html.jinja', result='No file provided')
    
    modelStr = request.form.get('model')
    if (modelStr == '1'): model = model1
    elif (modelStr == '2'): model = model2
    elif (modelStr == '3'): model = model3
    else: return render_template('results.html.jinja', result='No model selected')
    
    if file:
        try:
            img = Image.open(file).convert('L')
            img.save('static/temp.png')
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                probabilities = nn.functional.softmax(output, dim=1)
                confidence_scores = probabilities.numpy().flatten()
                prediction = np.argmax(confidence_scores)
            return render_template('results.html.jinja', 
                                   result=prediction, 
                                   predictions=confidence_scores.tolist(), 
                                   image_path='static/temp.png',
                                   model=modelStr,
                                   enumerate=enumerate)
        except Exception as e:
            print(f'Error during prediction: {e}')
            return render_template('results.html.jinja', result=f'Error during prediction: {e}')