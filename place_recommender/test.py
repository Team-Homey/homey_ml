import io
import requests
from flask import Flask, request, jsonify
from function.feature_extraction import *
from function.preprocessing import *
from function.similarity import *
import googlemaps
feature_model = FeatureExtractor()

test = [{"title": "Restaurant", "address": "Seoul, South Korea", "picture": "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=AUjq9jlychiaIDAjFyqurFROnXnTdJfNV3R5ZDKC2MguhRLA0jAlCM9WjU8pAwD48UHNTy5QUqr3MikFxygQyzblMJkotI7jsu5eGG2iDhEZbovuwEGAmVCaVD1tzz_rLNj57gtJRq3nX48Gqd7EJemDirNlwEVexFnpRAgDCZVQtDgDeZYs&key=AIzaSyAj6YPcapmjY-QjpThqYuhYIAoA84b6Oak", "url": "https://www.google.com/maps/search/?api=1&query=Google&query_place_id=ChIJcXYVN-GvfDURMfOCKkl4hB0"}, {"title": "Pizzeria O", "address": "86 Dongsung-gil, Jongno-gu", "picture": "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=AUjq9jldbyiMWeV_qCy4sg1qNlnyu7lG3FCF_UAgpHGbnpk5VxWNS--CB8pujzAWuxQO73S8Nku5ddV927vGPg1CbcDCN0BhP6IgmSe7DzUSX6U367i96RWqDlUojBkstCpo6I7C_wfZ-4erhY0IwbyuPX1URDwkrAzgFBYloETbUxOopo7p&key=AIzaSyAj6YPcapmjY-QjpThqYuhYIAoA84b6Oak", "url": "https://www.google.com/maps/search/?api=1&query=Google&query_place_id=ChIJ9Ye9LyyjfDURgYB4uqsmohw"}, {"title": "\uc608\ub78c\ub18d\uc6d0", "address": "649 \ub355\uc774\ub3d9 Ilsanseo-gu, Goyang-si", "picture": "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=AUjq9jlniH2yJvWe6C98redIzTwdMDumoYZ_ayScjn1dN7zUp5jL2z_xYLIDKLWJI-Nbhb5AYmeVrPmVbVavyLQh6tkDS2wWwGTt1jToKREIOkF6EkSLVxNcXJ63ZMdfISGfCO3_rWLTSMujeigjLWdCl1sW7xPpiai_6Q3gfg8MTmO8b9vB&key=AIzaSyAj6YPcapmjY-QjpThqYuhYIAoA84b6Oak", "url": "https://www.google.com/maps/search/?api=1&query=Google&query_place_id=ChIJrZqcvtGPfDURv14-oq2cnw4"}, {"title": "Coffee Shop", "address": "Oryu-dong, Seo-gu", "picture": "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=AUjq9jlaFHp13wOABu1TAfyFCO1c9ln6dCoZDdrG7YD_6fyD9No5eGmHfCNnF78fB6pKPVDWO_G1tleQ6C2S063n53MAYUQQqkAcyOt1sMJMm035L_jThM2wQaFT7YVnuuMz8LuyBa1QPuXj5cXaTAcFY-pai0HJewNorZDEwwGwSXzznBmi&key=AIzaSyAj6YPcapmjY-QjpThqYuhYIAoA84b6Oak", "url": "https://www.google.com/maps/search/?api=1&query=Google&query_place_id=ChIJ14fuavuBfDURW0l9Gi-UiL4"}]

for a in test :
    for key in a.keys():
        print(key, a[key])
    print('----------------------')
# import requests

# print("아아")

# localUrl = "http://127.0.0.1:8080/ping"
# cloudServerUrl = "http://34.22.71.66/ping"
# response = requests.get(cloudServerUrl)

# print(response.text)

# # Todo
# # 1. GET - /family/ids -> 가족 id 모두 가져오기
# # 3. 모든 가족에 대해서 다음을 반복
# #   3-1. GET - /photo/family/{id} -> 가족 갤러리 이미지 모두 가져오기
# #   3-2. 모델 돌려서 추천 정보 만들기
# #   3-3. 원하는 형태로 가공
# #   3-4. POST - /recommended-content/family-id/{id} -> 추천 정보 저장하기

# response = requests.get("http://34.22.71.66/family/ids")
# print(response.json()["familyIds"])
# familyIdList = response.json()["familyIds"];


# for x in familyIdList :
#     photoResponse = requests.get("http://34.22.71.66/photo/family/" + str(x))
#     photoResponse.encoding = 'UTF-8'
#     print("-----------{}------------".format(x))
#     print(photoResponse.json())

#     # Todo : 모델 돌려서 추천 정보 json formatting
#     resultJson = {
# 	    "title" : "test title",
# 	    "address" : "test address",
# 	    "picture" : "https://1234514",
# 	    "url" : "https://rdgd"
#     }

#     recommendedContentPostResponse = requests.post("http://34.22.71.66/recommended-content/family-id/" + str(x), json=resultJson)
#     print(recommendedContentPostResponse.status_code)   

# Flask 서버가 구동된 상태에서 http://localhost:5000/family/ids 등의 API endpoint를 호출하면 해당 API endpoint의 함수가 실행됩니다. 이때, Flask는 해당 함수에서 반환하는 JSON 데이터를 HTTP 응답으로 변환하여 클라이언트에게 반환합니다.

# 단, generate_recommendations 함수는 클라이언트에서 호출하는 것이 아니라 서버에서 주기적으로 실행되어야 합니다. 따라서 이 함수는 별도의 스케줄링 작업 등을 통해 주기적으로 호출되도록 설정해야 합니다.



# from flask import Flask, jsonify, request
# import requests

# app = Flask(__name__)
# cloud_server_url = "http://34.22.71.66"

# # GET - /family/ids -> 서버에 저장된 가족 id 모두 가져오기
# @app.route('/family/ids', methods=['GET'])
# def get_family_ids():
#     response = requests.get(cloud_server_url + "/family/ids")
#     return jsonify(response.json())

# # GET - /photo/family/{id} -> 서버에 저장된 가족 갤러리 이미지 모두 가져오기
# @app.route('/photo/family/<id>', methods=['GET'])
# def get_family_photos(id):
#     response = requests.get(cloud_server_url + "/photo/family/" + str(id))
#     return jsonify(response.json())

# # POST - /recommended-content/family-id/{id} -> 추천 정보 서버에 저장하기
# @app.route('/recommended-content/family-id/<id>', methods=['POST'])
# def save_recommended_content(id):
#     content = request.json
#     response = requests.post(cloud_server_url + "/recommended-content/family-id/" + str(id), json=content)
#     return jsonify({"status": response.status_code})

# # 모든 가족에 대해서 추천 정보 생성 후 저장
# @app.route('/generate-recommendations', methods=['POST'])
# def generate_recommendations():
#     family_id_list = requests.get(cloud_server_url + "/family/ids").json()['familyIds']

#     for family_id in family_id_list:
#         photos = requests.get(cloud_server_url + "/photo/family/" + str(family_id)).json()

#         # Todo : 모델 돌려서 추천 정보 json formatting
#         result_json = [
#             {
#                 "title": "test title",
#                 "address": "test address",
#                 "picture": "https://1234514",
#                 "url": "https://rdgd"
#             },
#             {
#                 "title": "test title3",
#                 "address": "test address",
#                 "picture": "https://1234514",
#                 "url": "https://rdgd"
#             },
#             {
#                 "title": "test title2",
#                 "address": "test address",
#                 "picture": "https://1234514",
#                 "url": "https://rdgd"
#             }
#         ]

#         recommended_content_post_response = requests.post(cloud_server_url + "/recommended-content/family-id/" + str(family_id), json=result_json)

#     return jsonify({"status": "success"})


# if __name__ == '__main__':
#     app.run(debug=True)
