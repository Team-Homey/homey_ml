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

user_address = '용산구 한강로동'
image_urls = [
		"https://storage.googleapis.com/homey-test-storage/0f6c85f6-84cc-41de-a3dc-c691a4a54a9d"
	]
class PlaceInfo():
    def __init__(self, title: str, address: str, picture: str, url: str):
        self.title = title
        self.address = address
        self.picture = picture
        self.url = url

@app.route('/')
def predict():
    input_imgs = list()
    top3_tensors = list()
    top3_classes = list()
    results = list()
        
    for url in image_urls:
        img_bytes = requests.get(url).content
        print(type(img_bytes))
        img = transform_image(img_bytes)
        input_imgs.append(feature_model.forward(img))
    top3_tensors = recommender.find_top3_similar(input_imgs)
    for output in top3_tensors:
        classes, probs = output_model.prediction(output)
        top3_classes.append(classes)
    top3_classes = list(set(top3_classes))
        
    location = gmaps.geocode(user_address)[0]['geometry']['location']
    cut = max(0, 4 - len(top3_classes)) if len(top3_classes) < 3 else 1
            
    for idx, place in enumerate(top3_classes) :
        places = gmaps.places_nearby(keyword = place,
                                     location = location,
                                     radius = 50000,
                                     rank_by = 'prominence')
        for place in places['results'][:cut]:
            search_url = 'https://www.google.com/maps/search/?api=1&query=Google&query_place_id=' + place['place_id']
            if 'photos' in place :
                photo_ref = place['photos'][0]['photo_reference']
                photo_url = 'https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={}&key={}'.format(photo_ref, api_key)
            else :
                photo_url = ''
                
            place_info = PlaceInfo(title=place['name'],
                                   address=place['vicinity'],
                                   picture=photo_url,
                                   url=search_url)
            results.append(place_info)
    return jsonify([vars(place) for place in results])

if __name__ == '__main__':
    app.run(debug=True)