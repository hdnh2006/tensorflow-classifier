#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:16:57 2023

@author: henry
"""


import tensorflow as tf
import wandb
import logging
import os
from pathlib import Path


def load_model(wandb_session, path='hdnh2006/bird_classifier/model_w5cxp2z2:v0'):
    """
    Load a TensorFlow model from a given path or from W&B artifacts.
    
    Parameters:
    - path (str): Local path to the model or W&B artifact path. 
                  Default is 'hdnh2006/bird_classifier/model_w5cxp2z2:v0'.
    
    Returns:
    - tf.function: The inference function associated with the loaded model.
    
    Raises:
    - Exception: If the model cannot be loaded.
    """
    
    # Be sure is not a url
    path = str(path)
    assert not path.lower().startswith(('http://', 'https://')), 'Sources available for weights: WandB and local path, https sources are not allowed'
    
    logging.info(f"Attempting to load model from {path}.")
    
    if not os.path.exists(path):
        try:
            logging.info("Path doesn't exist locally. Trying to fetch from W&B artifacts.")
            artifact = wandb_session.use_artifact(path, type='model')
            if not os.path.exists(artifact.file()):
                artifact_dir = artifact.download()
                logging.info(f"Artifact downloaded to {artifact_dir}.")
            else:
                logging.info(f'Artifact founded locally in {artifact.file()}.')
                artifact_dir = artifact.file()
                artifact_dir = str(Path(artifact_dir).parents[0])
            
        except Exception as e:
            logging.error(f"Error fetching model artifact from W&B. Error: {e}")
            raise Exception(f"Failed to fetch model from {path}.")
    else:
        artifact_dir = path
    
    try:
        # Load the model
        loaded = tf.saved_model.load(artifact_dir)
        logging.info(f"Model loaded successfully from {artifact_dir}.")
    except Exception as e:
        logging.error(f"Failed to load model from {artifact_dir}. Error: {e}")
        raise Exception(f"Failed to load model from {artifact_dir}.")
    
    return loaded.signatures['default']

    
    
    