# Example using Flask
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
import numpy as np
import io

app = Flask(__name__)
model = load_model('../trained_models/waste_sorter_model.h5')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] # Adjust based on your dataset

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return jsonify({'prediction': predicted_class_name, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)