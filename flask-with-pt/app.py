# ---------------
# MAIN FLASK SITE
# ---------------

# Imports
from flask import Flask, render_template, request
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import os
from simplexai.explainers.simplex import Simplex
from torch.utils.data import DataLoader, Subset
from latent_model_classes import MNISTModel1, MNISTModel2, MNISTModel3

app = Flask(__name__)

if not os.path.exists('static'): 
    os.makedirs('static')

# ------
# MODELS
# ------
# Load models
model1 = MNISTModel1()
model1_loaded = False

model2 = MNISTModel2()
model2_loaded = False

model3 = MNISTModel3()
model3_loaded = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Transform Function
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset for corpus
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
corpus_indices = np.random.choice(len(mnist_train), size=100, replace=False)
corpus_dataset = Subset(mnist_train, corpus_indices)
corpus_loader = DataLoader(corpus_dataset, batch_size=100, shuffle=False)

corpus_inputs, _ = next(iter(corpus_loader))
corpus_inputs = corpus_inputs.to(device)

# -----
# FLASK
# -----
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.post('/predict')
def predict():
    global model1_loaded, model2_loaded, model3_loaded

    # Check for uploaded image
    if 'file' not in request.files: return render_template('results.html.jinja', result='No file provided')

    file = request.files['file']
    if file.filename == '': return render_template('results.html.jinja', result='No file provided')
    
    # Load appropriate model
    # Only load a model if it has been selected and hasn't already been loaded
    modelStr = request.form.get('model')
    if (modelStr == '1'):
        if (not model1_loaded):
            model1.load_state_dict(torch.load('models/model1_latent.pth', weights_only=True))
            model1.eval()
            model1_loaded = True
        model = model1.to(device)
    elif (modelStr == '2'):
        if (not model2_loaded):
            model2.load_state_dict(torch.load('models/model2_latent.pth', weights_only=True))
            model2.eval()
            model2_loaded = True
        model = model2.to(device)
    elif (modelStr == '3'):
        if (not model3_loaded):
            model3.load_state_dict(torch.load('models/model3_latent.pth', weights_only=True))
            model3.eval()
            model3_loaded = True
        model = model3.to(device)
    else: return render_template('results.html.jinja', result='No model selected')
    
    if file:
        try:
            # Temp save image for later use
            img = Image.open(file).convert('L')
            img.save('static/temp.png')
            img = transform(img).unsqueeze(0).to(device)
            
            # Feed image into selected model
            output = model(img)
            probabilities = nn.functional.softmax(output, dim=1)
            confidence_scores = probabilities.cpu().detach().numpy().flatten()
            prediction = np.argmax(confidence_scores)
            
            # Compute latent representation
            img_latent = model.latent_representation(img).detach()
            corpus_latents = model.latent_representation(corpus_inputs).detach()
            
            # Apply Simplex
            simplex = Simplex(corpus_examples=corpus_inputs, corpus_latent_reps=corpus_latents)
            simplex.fit(test_examples=img, test_latent_reps=img_latent, reg_factor=0)
            
            # Retrieve top corpus examples
            top_k_indices = simplex.weights[0].topk(k=3).indices.cpu().detach().numpy()
            top_examples = [corpus_inputs[idx] for idx in top_k_indices]
            
            # Save top examples as images
            top_example_paths = []
            for i, img_tensor in enumerate(top_examples):
                img_pil = Image.fromarray((img_tensor.cpu().detach().numpy().squeeze() * 255).astype(np.uint8))
                img_path = f'static/top_example_{i}.png'
                img_pil.save(img_path)
                top_example_paths.append(img_path)
                
            return render_template('results.html.jinja', 
                                   result=prediction, 
                                   predictions=[round(x, 4) for x in confidence_scores.tolist()], 
                                   image_path='static/temp.png',
                                   model=modelStr,
                                   enumerate=enumerate,
                                   top_indices=top_k_indices.tolist(),
                                   top_example_paths=top_example_paths)
        except Exception as e:
            # Something went wrong during image opening, prediction, or Simplex
            print(f'Error during prediction: {e}')
            return render_template('results.html.jinja', result=f'Error during prediction: {e}')