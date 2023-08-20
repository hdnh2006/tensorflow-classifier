# Deployment
This script launches a Flask application that serves an API endpoint for uploading images. The application uses pre-trained model to inference over the image.

Files
- `classify_api.py`: This script includes the Flask application.

# Usage

## 1. Install requirements
It is recommended to use a virtual environment to install the dependencies.
```
pip install -r requirements.txt
```

## 2. Launch the Flask Application
To start the application, run the `classify_api.py` script using Python:
```
python `deployment/classify_api.py` --device cpu --weights ...
```

You can specify the paths to the pre-trained model `--weights` option. If you don't provide these options, the script will use default paths.

The application will start a local server where you can access the API. The exact URL will be printed in the console when the application starts, but it's usually `http://localhost:5000/`.


## 3. Local testing inside specific folder
Just open your favorite browser and go to http://0.0.0.0:5000/classify?source=data/images will inference all the images inside the folder `data/images`.

The API will return the images labeled.

## 4 Call from terminal or python program
The `client.py` code provides several example about how the API can be called. A very common way to do it is to call a public image from url and to get the coordinates of the bounding boxes:

```python
import requests

api_url = 'http://172.17.0.2:5000/classify'
resp = requests.get(f'{api_url}?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg&save_labels=T', verify=False)

```
And you will get a json with the following data:

```
b'{"results": [{"conf": 0.96, "class": "Microcarbo melanoleucos"}, {"conf": 0.0, "class": "Piranga rubra"}, {"conf": 0.0, "class": "Piranga olivacea"}]}'
```

## 5. Public images

You can provide a public image to the endpoint. The ling should end in `.jpg, .png, .jpeg, ...` or other common image formats as follows:
```
http://0.0.0.0:5000/classify?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg
# or
http://0.0.0.0:5000/classify?source=https://www.lyricbirdfood.com/media/1880/summer-tananger.jpg&save_labels=T # it will return a json
```

## 6. Recommendation
For the correct functioning of the application, use `client.py` from any IDE or terminal.
