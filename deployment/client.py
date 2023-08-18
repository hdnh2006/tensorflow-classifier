#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   ____   _   _                  _                       
  / ___| | | (_)   ___   _ __   | |_       _ __    _   _ 
 | |     | | | |  / _ \ | '_ \  | __|     | '_ \  | | | |
 | |___  | | | | |  __/ | | | | | |_   _  | |_) | | |_| |
  \____| |_| |_|  \___| |_| |_|  \__| (_) | .__/   \__, |
                                          |_|      |___/ 
                                          
The following lines of code show how to make requests to the API
"""



import requests

api_url = 'http://172.17.0.2:5000/classify'


# ====================== Public image ====================== #

# Saving txt file


resp = requests.get(f'{api_url}?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg&save_labels=T',
                    verify=False)
print(resp.content)
# b'{"results": [{"conf": 0.96, "class": "Microcarbo melanoleucos"}, {"conf": 0.0, "class": "Piranga rubra"}, {"conf": 0.0, "class": "Piranga olivacea"}]}'

# Without save txt file, just labeling the image
resp = requests.get(f'{api_url}?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg',
                    verify=False)
print(resp.content)

# You can also copy and paste the following url in your browser
print(f'{api_url}?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg')


# ============== Multiple images ======= #
image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
]

for image in image_urls:
    print(image)
    resp = requests.get(f'{api_url}?source={image}', verify=False)




# ============== Local image ================ #


# Define the path to your image and the API URL
image_path = 'data/images/679edc606d9a363f775dabf0497d31de8c3d7060.jpg'
api_url = f'{api_url}'

# Define the parameters you want to add to the JSON body
data = {
    'save_labels': 'T' # Set None in case you want the image in bytes
}

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Prepare the file for upload
    files = {'myfile': image_file}
    # Make the POST request to upload the image with the specified parameters in the JSON body
    response = requests.post(api_url, files=files, data=data)

print(response.content)
# b'{"results": [{"conf": 0.83, "class": "Alauda arvensis"}, {"conf": 0.09, "class": "Tringa semipalmata inornatus"}, {"conf": 0.02, "class": "Megaceryle alcyon"}]}'
