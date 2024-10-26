from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Preprocessing function
def preprocess_image(img):
    img = tf.image.resize(img, [28, 28])
    img = img / 255.0  # Normalize here
    img = tf.expand_dims(img, axis=0)  # Shape should be (1, 28, 28, 1)
    return 

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('results.html.jinja', result='No file provided')
    file = request.files['file']
    if file.filename == '':
        return render_template('results.html.jinja', result='No file provided')
    if file:
        from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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
            file_bytes = file.read()
            
            if not file_bytes:
                return render_template('results.html.jinja', result='File read failed')

            image_upload = tf.image.decode_image(file_bytes, channels=1, dtype=tf.float32)
            
            if image_upload is None:
                return render_template('results.html.jinja', result='Image decode failed')
            
            image_upload = preprocess_image(image_upload)

            model = tf.keras.models.load_model('model.keras')

            prediction = model.predict(image_upload)
            predicted_digit = np.argmax(prediction, axis=1)[0]

            print(f'Predicted digit: {predicted_digit}')

            return render_template('results.html.jinja', result=predicted_digit)
        except Exception as e:
            print(f'Error during prediction: {e}')
            return render_template('results.html.jinja', result=f'Error during prediction: {e}')
        
def preprocess_image(img):
    img = tf.image.resize(img, [28, 28])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img