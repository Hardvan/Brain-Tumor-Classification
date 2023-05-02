from flask import Flask, render_template, request
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io

# Suppress Tensorflow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():

    model = load_model("./brainT_detect1682946468.3700671.h5")

    prediction = None

    if request.method == 'POST':

        # Get the uploaded file
        f = request.files['fileUpload']

        # If the file exists and is allowed, make a prediction
        if f and allowed_file(f.filename):

            # Load the image and preprocess it for the model
            img = image.load_img(io.BytesIO(f.read()), target_size=(180, 180))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Make a prediction using the loaded model
            class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            prediction = model.predict(x)
            score = tf.nn.softmax(prediction[0])

            tumor_name = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            # Read the image file and convert it to base64 encoding
            f.seek(0)
            image_file = f.read()
            image_base64 = base64.b64encode(image_file).decode('utf-8')

            # Set the prediction value to be displayed in the HTML template
            result = {'image': image_base64,
                      'tumor_name': tumor_name,
                      'confidence': confidence}

            return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
