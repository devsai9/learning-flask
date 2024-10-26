# Learning Flask - MNIST Digit Identification [PyTorch]
This is a project I made to help me learn Flask and how to deploy machine learning models using Flask. The project uses a neural network with the PyTorch Python package to identify digits from the MNIST dataset.

## Installation
1. Clone the repository
2. Open the termianl
3. Create a virtual environment
4. Start up the virtual environment
5. Install dependencies using `pip install -r requirements.txt`
6. Run the app using `flask run`

After doing these steps, you can visit http://127.0.0.1:5000/ in your browser and try uploading some of the picture in the `human_test_images` folder.

## Details
I used PyTorch to create this. <br>
The model is a neural network that looks like this: <br>
```python
self.model = Sequential(
    Flatten(),
    Linear(28 * 28, 512),
    ReLU(),
    Linear(512, 512),
    ReLU(),
    Linear(512, 10),
)
```
The model was trained and stored in a `model.pth` file.