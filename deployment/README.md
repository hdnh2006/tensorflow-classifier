# Deployment
This script launches a Flask application that serves an API endpoint for uploading CSV files containing transaction data. The application uses pre-trained XGBoost and autoencoder models to predict the likelihood of each transaction being fraudulent.

Files
- `app.py`: This script includes the Flask application.

# Usage

1. Install requirements
It is recommended to use a virtual environment to install the dependencies.
```
pip install -r requirements.txt
```

2. Launch the Flask Application
To start the application, run the `app.py` script using Python:
```
python app.py --xgboost path_to_xgboost_model --autoencoder path_to_autoencoder_model
```

You can specify the paths to the pre-trained models using the `--xgboost` and `--autoencoder` options. If you don't provide these options, the script will use default paths.

The application will start a local server where you can access the API. The exact URL will be printed in the console when the application starts, but it's usually `http://localhost:5000/`.

# API Endpoint
The API provides a single endpoint at the main, which accepts both GET and POST requests. When you make a GET request, the application will return an HTML form where you can upload a CSV file.

When you make a POST request and include a CSV file in the file field of the request, the application will process the file, make predictions using the pre-trained models, and return a CSV file containing the predictions. The returned CSV file includes two columns: Autoencoder Predictions and XGBoost Predictions, which contain the scores from the autoencoder and XGBoost models respectively.

For the correct functioning of the application, it is important that the data in the uploaded CSV file has the same format as the data that the models were trained on.