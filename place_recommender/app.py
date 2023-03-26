import io
import json
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
cloud_server_url = "http://34.22.71.66"

# class PlaceInfo:
#     def __init__(self, title, address, picture, url):
#         self.title = title
#         self.address = address
#         self.picture = picture
#         self.url = url

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
    cut = max(0, 3 - len(top3_classes)) if len(top3_classes) < 3 else 1
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

            place_info = {
                'title' : place['name'],
                'address' : place['vicinity'],
                'picture' : photo_url,
                'url' : search_url
            }
            results.append(place_info)
    return results

# GET - /family/ids -> 서버에 저장된 가족 id 모두 가져오기
@app.route('/family/ids', methods=['GET'])
def get_family_ids():
    response = requests.get(cloud_server_url + "/family/ids")
    return response.json()["familyIds"]

# GET - /photo/family/{id} -> 서버에 저장된 가족 갤러리 이미지 모두 가져오기
@app.route('/photo/family/<id>', methods=['GET'])
def get_family_info(id):
    response = requests.get(cloud_server_url + "/photo/family/" + str(id))
    response.encoding = 'UTF-8'
    return response.json()['address'], response.json()['images']

@app.route('/recommended-content/family-id/<id>', methods=['POST'])
def generate_recommendations():
    family_ids = get_family_ids()
    for family_id in family_ids: 
        address, image_urls = get_family_info(family_id)

        # Run recommender
        top3_tensors = get_top3_similar_images(image_urls)
        top3_classes = get_top3_classes(top3_tensors)
        nearby_places = get_nearby_places(top3_classes, address)
        result_json = json.dumps(nearby_places) 
               
        response = requests.post(cloud_server_url + "/recommended-content/family-id/" + str(family_id), json=result_json)

        return result_json, jsonify({"status": response.status_code})
        
if __name__ == '__main__':
    app.run(debug=True)