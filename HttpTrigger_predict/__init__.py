import logging
import os
import sys
import joblib
from azure.storage.blob import BlobServiceClient
import azure.functions as func
import pandas as pd
import numpy as np
import scipy.sparse as sp
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collaborative_filtering import CollaborativeFiltering

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Log the request method and parameters
    logging.info(f"Request method: {req.method}")
    logging.info(f"Request params: {req.params}")

    blob_service_client = BlobServiceClient.from_connection_string('AZURE_STORAGE_CONNECTION_STRING')
    container_name = "projet09"

    try:
        # Télécharger user_article_matrix.csv
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="user_article_matrix.csv")
        matrix_stream = blob_client.download_blob().readall()
        user_article_matrix = pd.read_csv(io.BytesIO(matrix_stream))
        logging.info(f"user_article_matrix shape: {user_article_matrix.shape}")
        logging.info(f"user_article_matrix columns: {user_article_matrix.columns}")
        
        # Télécharger svd_model.joblib
        blob_client = blob_service_client.get_blob_client(container=container_name, blob="svd_model.joblib")
        model_stream = blob_client.download_blob().readall()
        model = joblib.load(io.BytesIO(model_stream))
        logging.info("Model loaded successfully")

        # Initialisation de la classe CollaborativeFiltering
        cf = CollaborativeFiltering(None, None)
        cf.load_model_df(model)
        cf.load_user_article_matrix_df(user_article_matrix)
        logging.info("CollaborativeFiltering class initialized successfully")

    except Exception as e:
        logging.error(f"Failed to load resources or initialize class: {str(e)}")
        return func.HttpResponse(
            f"Failed to load resources or initialize class: {str(e)}",
            status_code=500
        )

    # Prédiction pour un user_id donné
    user_id_param = req.params.get('user_id')
    logging.info(f"Received user_id: {user_id_param}")
    
    if user_id_param:
        try:
            user_id = int(user_id_param)
            logging.info(f"Converted user_id to int: {user_id}")
            recommended_articles = cf.recommend_articles(user_id, top_n=5)
            logging.info(f"Recommended articles: {recommended_articles}")
            return func.HttpResponse(str(recommended_articles), status_code=200)
        except ValueError as ve:
            logging.error(f"Invalid user_id passed: {user_id_param} is not an integer. Error: {ve}")
            return func.HttpResponse(f"Invalid user_id. Please ensure it is a valid integer. Error: {ve}", status_code=400)
        except Exception as e:
            logging.error(f"Error in recommendation process: {str(e)}")
            return func.HttpResponse(f"Error in recommendation process: {str(e)}", status_code=500)
    else:
        logging.error("No user_id provided in the request")
        return func.HttpResponse("Please pass a user_id in the query string for personalized article recommendations.", status_code=400)
