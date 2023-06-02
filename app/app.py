# =[Modules dan Packages]========================

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten

# Membuat instance Flask
app = Flask(__name__, static_url_path='/static')

# membuat model model
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

# memuat model
model.load_weights('model_detect2.h5')
print("Model loaded successfully.")

# Fungsi prediksi
def predict_disease(chili_plant):
    test_image = load_img(chili_plant, target_size=(224, 224))
    print("@@ Image for prediction")

    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    print(pred)

    if pred == 0:
        return "Chili - Antracnose", 'Antracnose.html'
    elif pred == 1:
        return "Chili - Healthy", 'Healthy.html'
    elif pred == 2:
        return "Chili - Leaf_spot", 'Leaf_spot.html'
    elif pred == 3:
        return "Chili - Leaf_crul", 'Leaf_crul.html'
    elif pred == 4:
        return "Chili - Whitely", 'Whitely.html'
    elif pred == 5:
        return "Chili - Yellowwish", 'Yellowwish.html'

    return None, None


# Routing  halaman utama Home
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

#Routing 
@app.route("/predict", methods=['POST'])
def predict():
    #Prediksi hasil gambar 
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='Please select an image to upload.')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='Please select an image to upload.')

        filename = file.filename
        print("@@ Input post =", filename)

        file_path = os.path.join('static/uploads/', filename)
        file.save(file_path)

        print("@@ Predict class...")
        pred, output_page = predict_disease(chili_plant=file_path)
        
        if output_page is None:
            return render_template('index.html', error='Invalid image. Please upload an image of chili plant.')
        
        return render_template(output_page, pred_output=pred, user_image=file_path)


# =[Main]========================================		
if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
