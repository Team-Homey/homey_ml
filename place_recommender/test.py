@app.route('/', methods=['GET'])
def predict():
    input_imgs = []
    top3_tensors = []
    top3_classes = []
    results = []

    # 주소와 이미지 URL을 GET 요청에서 가져옵니다
    address = request.args.get('address')
    image_urls = request.args.getlist('images')

    # 이미지 URL을 이용하여 이미지를 로드하고 변환합니다
    for url in image_urls:
        img_bytes = requests.get(url).content
        img = transform_image(img_bytes)
        input_imgs.append(feature_model.forward(img))

    top3_tensors = recommender.find_top3_similar(input_imgs)
    for output in top3_tensors:
        classes, probs = output_model.prediction(output)
        top3_classes.append(classes)
    top3_classes = list(set(top3_classes))

    location = gmaps.geocode(address)[0]['geometry']['location']
    cut = max(0, 4 - len(top3_classes)) if len(top3_classes) < 3 else 1

    for idx, place in enumerate(top3_classes):
        places = gmaps.places_nearby(keyword=place,
                                     location=location,
                                     radius=50000,
                                     rank_by='prominence')

        for place in places['results'][:cut]:
            search_url = 'https://www.google.com/maps/search/?api=1&query=Google&query_place_id=' + place['place_id']
            photo_url = 'https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={}&key={}'.format(place['photos'][0]['photo_reference'], api_key)
            place_info = PlaceInfo(title=place['name'],
                                   address=place['vicinity'],
                                   picture=photo_url,
                                   url=search_url)
            results.append(place_info)
    return jsonify([vars(place) for place in results])


import io
import requests
import googlemaps
from flask import Flask, request, jsonify
from function.feature_extraction import FeatureExtractor
from function.preprocessing import transform_image
from function.similarity import ComputeSimilarity

app = Flask(__name__)

FEATURE_MODEL = FeatureExtractor()
OUTPUT_MODEL = FeatureExtractor()
RECOMMENDER = ComputeSimilarity()
API_KEY = 'YOUR_API_KEY'
GMAPS = googlemaps.Client(API_KEY)

USER_ADDRESS = '용산구 한강로동'
IMAGE_URLS = [
    "https://storage.googleapis.com/homey-test-storage/0f6c85f6-84cc-41de-a3dc-c691a4a54a9d"
]

class PlaceInfo():
    def __init__(self, name: str, address: str, photo_url: str, search_url: str):
        self.name = name
        self.address = address
        self.photo_url = photo_url
        self.search_url = search_url

@app.route('/')
def predict():
    input_images = []
    top3_tensors = []
    top3_classes = []
    results = []
        
    for image_url in IMAGE_URLS:
        image_bytes = requests.get(image_url).content
        image = transform_image(image_bytes)
        input_images.append(FEATURE_MODEL.forward(image))
    
    top3_tensors = RECOMMENDER.find_top3_similar(input_images)
    for tensor in top3_tensors:
        classes, probabilities = OUTPUT_MODEL.prediction(tensor
