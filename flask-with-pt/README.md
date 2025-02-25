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
I used PyTorch to create three neural networks models and Flask to host a intuitive application for users to use the models.<br>
Each model was trained and stored in a corresponding `model[n]_latent.pth` file.<br>
More model statistics can be found on the website `/stats` page of the Flask website.

### Simplex
In [Commit 97e8b02](https://github.com/devsai9/learning-flask/commit/97e8b0210a2d6dbbd4e9db6497545fb1e5c186d7), [Simplex](https://github.com/JonathanCrabbe/Simplex) was added to the project.<br>
Simplex is a library that uses its own `explainer`s to explain the predictions of a model.<br>

In the next major commit ([Commit ff75692](https://github.com/devsai9/learning-flask/commit/ff75692802ce533790e5d56bec7edb8de86ac64f)), Simplex's functionality was extended to the existing Flask website.<br>
This allows the user to see the top three images that explains the model's prediction.