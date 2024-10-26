# Learning Flask - MNIST Digit Identification [TensorFlow]
This is a project I made to help me learn Flask and how to deploy machine learning models using Flask. The project uses a neural network with the TensorFlow Python package to identify digits from the MNIST dataset.

## Installation
1. Clone the repository
2. Open the termianl
3. Create a virtual environment
4. Start up the virtual environment
5. Install dependencies using `pip install -r requirements.txt`
6. Run the app using `flask run`

After doing these steps, you can visit http://127.0.0.1:5000/ in your browser and try uploading some of the picture in the `human_test_images` folder.

## Details
I used Tensorflow to create this. <br>
The model is a neural network that looks like this: <br>
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(784, activation='relu'),
    Dense(392, activation='relu'),
    Dense(196, activation='relu'),
    Dense(98, activation='relu'),
    Dense(10, activation='softmax')
])
```
The model was trained and stored in a `model.keras` file.