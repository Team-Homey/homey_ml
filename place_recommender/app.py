import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from flask import Flask, jsonify
from PIL import Image

from function.feature_extraction import *
from function.preprocessing import *
from function.similarity import *

app = Flask(__name__)
    
feature_model = FeatureExtractor()
output_model = FeatureExtractor()
recommender = ComputeSimilarity()
input_imgs = list()
top3_tensors = list()
top3_classes = list()

@app.route('/')
def predict():
    with open("data/12.jpg", 'rb') as f :
        img_bytes = f.read()
        img = transform_image(img_bytes)
        input_imgs.append(feature_model.forward(img))
        top3_tensors = recommender.find_top3_similar(input_imgs)
        for output in top3_tensors:
            classes, probs = output_model.prediction(output)
            top3_classes.append(classes)
        return jsonify({'top1_class_name' : top3_classes[0], 
                        'top2_class_name' : top3_classes[1],
                        'top3_class_name' : top3_classes[2]
                        })
        

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.methods = 'POST' :
#         file = request.files['file']
#         img_bytes = f.read()
#         probs, idx = get_prediction(img_bytes)
#         return jsonify({'top1_class_name' : top3_classes[0], 
#                         'top2_class_name' : top3_classes[1],
#                         'top3_class_name' : top3_classes[2]
#                         })



# set FLASK_ENV=development 
# set FLASK_APP=app
# set FLASK_DEBUG=true 
# flask run
