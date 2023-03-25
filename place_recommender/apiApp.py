import requests
import app
print("아아")

localUrl = "http://127.0.0.1:8080/ping"
cloudServerUrl = "http://34.22.71.66/ping"
response = requests.get(cloudServerUrl)

print(response.text)

# Todo
# 1. GET - /family/ids -> 가족 id 모두 가져오기
# 3. 모든 가족에 대해서 다음을 반복
#   3-1. GET - /photo/family/{id} -> 가족 갤러리 이미지 모두 가져오기
#   3-2. 모델 돌려서 추천 정보 만들기
#   3-3. 원하는 형태로 가공
#   3-4. POST - /recommended-content/family-id/{id} -> 추천 정보 저장하기

response = requests.get("http://34.22.71.66/family/ids")
print(response.json()["familyIds"])
familyIdList = response.json()["familyIds"]


for x in familyIdList :
    photoResponse = requests.get("http://34.22.71.66/photo/family/" + str(x))
    photoResponse.encoding = 'UTF-8'
    
    print(photoResponse.json())

    # Todo : 모델 돌려서 추천 정보 json formatting
    resultJson = app.predict(photoResponse.json()["images"])

    # resultJson = {
	#     "title" : "test title",
	#     "address" : "test address",
	#     "picture" : "https://1234514",
	#     "url" : "https://rdgd"
    # }

    recommendedContentPostResponse = requests.post("http://34.22.71.66/recommended-content/family-id/" + str(x), json=resultJson)
    print(recommendedContentPostResponse.status_code)   


