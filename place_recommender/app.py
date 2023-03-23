import io
import requests
from flask import Flask, request, jsonify
from function.feature_extraction import *
from function.preprocessing import *
from function.similarity import *
import googlemaps

app = Flask(__name__)
    
feature_model = FeatureExtractor()
output_model = FeatureExtractor()
recommender = ComputeSimilarity()
api_key = 'AIzaSyAj6YPcapmjY-QjpThqYuhYIAoA84b6Oak'
gmaps = googlemaps.Client(api_key)

# user_adderss and image_urls should be changed when there exits a event for user to upload the image and change the user_address. 
user_address = '용산구 한강로동'
image_urls = [
    "https://storage.googleapis.com/homey-test-storage/0f6c85f6-84cc-41de-a3dc-c691a4a54a9d"
]

class PlaceInfo:
    def __init__(self, title, address, picture, url):
        self.title = title
        self.address = address
        self.picture = picture
        self.url = url

def download_image(url):
    img_bytes = requests.get(url).content
    img = transform_image(img_bytes)
    return feature_model.forward(img)

def get_top3_similar_images(image_urls):
    input_imgs = [download_image(url) for url in image_urls]
    return recommender.find_top3_similar(input_imgs)

def get_top3_classes(top3_tensors):
    top3_classes = []
    for output in top3_tensors:
        classes, probs = output_model.prediction(output)
        top3_classes.append(classes)
    top3_classes = list(set(top3_classes))
    return top3_classes

def get_nearby_places(top3_classes, user_address):
    results = []
    location = gmaps.geocode(user_address)[0]['geometry']['location']
    cut = max(0, 4 - len(top3_classes)) if len(top3_classes) < 3 else 1
    for idx, place in enumerate(top3_classes):
        places = gmaps.places_nearby(keyword=place,
                                     location=location,
                                     radius=50000,
                                     rank_by='prominence')
        for place in places['results'][:cut]:
            if 'photos' in place:
                photo_ref = place['photos'][0]['photo_reference']
                photo_url = f'https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={api_key}'
            else:
                photo_url = ''

            search_url = f'https://www.google.com/maps/search/?api=1&query=Google&query_place_id={place["place_id"]}'
            place_info = PlaceInfo(title=place['name'],
                                   address=place['vicinity'],
                                   picture=photo_url,
                                   url=search_url)
            results.append(place_info)
    return results

@app.route('/')
def predict():
    # image_urls = request.args.getlist('image_url')
    # user_address = request.args.get('user_address')
    input_imgs = [download_image(url) for url in image_urls]
    top3_tensors = get_top3_similar_images(image_urls)
    top3_classes = get_top3_classes(top3_tensors)
    nearby_places = get_nearby_places(top3_classes, user_address)
    
    return jsonify([vars(place) for place in nearby_places])

if __name__ == '__main__':
    app.run(debug=True)