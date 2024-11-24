import os
from flask import Flask, request, send_from_directory, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from models.skin_tone.skin_tone_knn import identify_skin_tone
from models.recommender.rec import recs_essentials, makeup_recommendation
from PIL import Image
import tensorflow as tf
import tf_keras as k3  # Import tf_keras instead of tensorflow.keras
import numpy as np
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__, static_folder="./build")
api = Api(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load models
class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'models/skin_tone/skin_tone_dataset.csv'

def get_model():
    global model1, model2
    model1 = k3.models.load_model('./models/skin_model')  # Use k3 to load model
    print('Model 1 loaded')
    model2 = k3.models.load_model('./models/acne_model')  # Use k3 to load model
    print("Model 2 loaded!")

def load_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

def prediction_skin(img_path):
    new_image = load_image(img_path)
    pred1 = model1.predict(new_image)
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    return pred_class1

def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2

get_model()

# API Endpoints
class SkinMetrics(Resource):
    def put(self):
        args = request.json
        file = args['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data + b'==')))
        filename = 'image.png'
        file_path = os.path.join('./static', filename)
        im.save(file_path)
        skin_type = prediction_skin(file_path).split('_')[0]
        acne_type = prediction_acne(file_path)
        tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)
        return {'type': skin_type, 'tone': str(tone), 'acne': acne_type}, 200

class Recommendation(Resource):
    def put(self):
        args = request.json
        features = args['features']
        tone = args['tone']
        skin_type = args['type'].lower()
        skin_tone = 'light to medium'
        if tone <= 2:
            skin_tone = 'fair to light'
        elif tone >= 4:
            skin_tone = 'medium to dark'
        fv = [int(value) for value in features.values()]
        general = recs_essentials(fv, None)
        makeup = makeup_recommendation(skin_tone, skin_type)
        return {'general': general, 'makeup': makeup}

# Serve React app
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

# Register API endpoints
api.add_resource(SkinMetrics, "/api/upload")
api.add_resource(Recommendation, "/api/recommend")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)
