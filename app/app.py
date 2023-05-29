from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten

# Create Flask instance
app = Flask(__name__, static_url_path='/static')

# Create the model
base_model = VGG16(input_shape=(224, 224, 3),
                   include_top=False,
                   weights='imagenet')

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(7, activation="softmax", name="classification"))

# Load the model weights
model.load_weights('model_detect1.h5')
print("Model weights loaded successfully.")

# Define the prediction function
def predict_disease(chili_plant):
    test_image = load_img(chili_plant, target_size=(224, 224))
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    print(pred)

    if pred[0] == 0:
        return "Antracnose", 'Antracnose.html'
    elif pred[0] == 1:
        return "Healthy", 'Healthy.html'
    elif pred[0] == 2:
        return "Leaf_spot", 'Leaf_spot.html'
    elif pred[0] == 3:
        return "Leaf_crul", 'Leaf_crul.html'
    elif pred[0] == 4:
        return "Whitely", 'Whitely.html'
    elif pred[0] == 5:
        return "Yellowwish", 'Yellowwish.html'
    else:
        return "Unknown", 'Unknown.html'


# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input image from client, then predict class and render respective .html page for solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted =", filename)
        
        file_path = os.path.join('app/static/uploads/', filename)
        file.save(file_path)

        print("@@ Predicting class...")
        pred, output_page = predict_disease(chili_plant=file_path)
              
        return render_template(output_page, pred_output=pred, user_image=file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
