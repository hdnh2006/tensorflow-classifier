# Deployment
This script launches a Flask application that serves an API endpoint for uploading images. The application uses pre-trained model to inference over the image.

Files
- `classify_app.py`: This script includes the Flask application.

# Usage

1. Install requirements
It is recommended to use a virtual environment to install the dependencies.
```
pip install -r requirements.txt
```

2. Launch the Flask Application
To start the application, run the `app.py` script using Python:
```
python `classify_app.py` --device cpu --weights ...
```

You can specify the paths to the pre-trained model `--weights` option. If you don't provide these options, the script will use default paths.

The application will start a local server where you can access the API. The exact URL will be printed in the console when the application starts, but it's usually `http://localhost:5000/`.

# API Endpoint
The API provides a single endpoint at the main, which accepts both GET and POST requests. When you make a GET request, the application will return an HTML form where you can upload an image file.

When you make a POST request and include a `save_labels=T` flag in the file field of the request, the application will process the file, make predictions using the pre-trained models, and return a json with the predictions. The returned json filee includes: Confidence, Class.

3 Recommendation
For the correct functioning of the application, use `client.py` from any IDE or terminal.
