#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:21:28 2023

@author: henry
"""

import logging
from pathlib import Path
import urllib
import os
import tensorflow.compat.v2 as tf
import time
import cv2
import json

def check_file(file, raw_path: Path = '.'):
    """
    Check if a file exists locally. If not, download it from the provided URL.
    
    Parameters:
    - file (str): Local file path or URL.
    
    Returns:
    - str: File path of the existing or downloaded file.
    
    Raises:
    - FileNotFoundError: If the provided path is neither a file nor a valid URL.
    - AssertionError: If the file fails to download from the URL.
    """
    file = str(file)  # convert to str()

    # If the file exists locally or no file is provided
    if os.path.isfile(file) or not file:
        return file

    # If a URL is provided
    elif file.startswith(('http:/', 'https:/')):
        url = file
        file = str(raw_path / Path(urllib.parse.unquote(file).split('?')[0]).name)  # Convert '%2F' to '/', and split "https://url.com/file.txt?auth"
        
        # If the file from the URL exists locally
        if os.path.isfile(file):
            logging.info(f'Found {url} locally at {file}')
            return file

        # If the file from the URL doesn't exist locally, download it
        logging.info(f'Downloading {url} to {file}...')
        urllib.request.urlretrieve(url, file)

        # Ensure the downloaded file exists and is not empty
        if not (Path(file).exists() and Path(file).stat().st_size > 0):
            logging.error(f'File download failed: {url}')
            raise AssertionError(f'File download failed: {url}')

        return file

    # If the provided path is neither a file nor a valid URL
    else:
        logging.error(f'File {file} does not exist and is not a valid URL.')
        raise FileNotFoundError(f'{file} does not exist')
        
        

def select_device(device='', batch_size=None):
    """
    Selects the appropriate device for TensorFlow based on the given input.
    
    Parameters:
    - device (str): Device identifier. Can be 'cpu', '0', or '0,1,2,3' etc.
                   Default is an empty string which means auto-detection.
    - batch_size (int, optional): Batch size for the operation. Currently not used in the function.
    
    Returns:
    - str: Device name string in TensorFlow format, e.g., '/CPU:0' or '/GPU:0'.
    
    Raises:
    - AssertionError: If CUDA is unavailable for the requested GPU device.
    """
    cpu = device.lower() == 'cpu'

    # Force TensorFlow to not see any GPUs
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Non-CPU device requested
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # Set environment variable

        # Check if any GPU is visible to TensorFlow
        if not len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            logging.error(f'CUDA unavailable, invalid device {device} requested')
            raise AssertionError(f'CUDA unavailable, invalid device {device} requested')

    # Determine if CUDA is available and set the device name accordingly
    cuda = not cpu and len(tf.config.experimental.list_physical_devices('GPU')) > 0
    device_name = f'/GPU:{device}' if cuda else '/CPU:0'

    logging.info(f"Using device: {device_name}")

    return device_name




def time_synchronized():
    """
    Provides a TensorFlow-accurate time. If GPU is available, it synchronizes 
    the GPU and CPU time by running a no-op TensorFlow operation on the GPU. 
    Otherwise, it simply returns the current time.
    
    Returns:
    - float: The current time in seconds since the epoch (same as time.time()).
    """
    # Check if GPU is available
    if tf.config.experimental.list_physical_devices('GPU'):
        with tf.device('/gpu:0'):
            tf.no_op()

    return time.time()




def increment_path(path):
    """
    Increment a file or directory path.
    
    Parameters:
    - path (str): The original file or directory path.
    
    Returns:
    - pathlib.Path: The incremented path.
    """
    path = Path(path)  # Make path OS-agnostic

    # If the path exists
    if path.exists():
        # Split the path into its base and suffix if it's a file
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Increment the path until a unique path is found
        for n in range(2, 9999):
            p = f'{path}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)

    return path





def auto_size_text(img, text, max_width_percent=0.8):
    """
    Determine the font scale to fit the text inside the image, starting from the top left.
    
    Args:
    - img: The target image (numpy array).
    - text: The string of text.
    - max_width_percent: Maximum width the text can occupy as a fraction of image width.
    
    Returns:
    - font_scale: The calculated font scale.
    - position: The calculated (x, y) position to start drawing the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    max_width = int(img.shape[1] * max_width_percent)
    
    # Splitting the text by newlines and calculating the width of the longest line
    lines = text.split('\n')
    max_text_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
    
    # Reduce font size until text width fits within allowable width
    while max_text_width > max_width:
        font_scale -= 0.1
        max_text_width = 0
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_text_width = max(max_text_width, text_width)
    
    start_x = 32
    _, text_height = cv2.getTextSize("Test", font, font_scale, thickness)
    
    return font_scale, (start_x, text_height + 32)



def annotate(img, text, txt_color=(255, 255, 255)):
    """
    Annotate an image with the given text, auto-adjusting the font scale and positioning.
    
    Parameters:
    - img (numpy.ndarray): The target image.
    - text (str): The string of text to be displayed on the image.
    - txt_color (tuple, optional): RGB color for the text. Default is white (255, 255, 255).
    
    Returns:
    None. The function modifies the input image in-place.
    """
        
    font_scale, (start_x, start_y) = auto_size_text(img, text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Split the text by newlines and draw each line
    for line in text.split('\n'):
        cv2.putText(img, line.strip(), (start_x, start_y), font, font_scale, txt_color, thickness)
        start_y += int(1.5 * cv2.getTextSize(line, font, font_scale, thickness)[0][1])  # 1.5 is the line spacing


def update_options(request):
    """
    Args:
    - request: Flask request object
    
    Returns:
    - source: URL string
    - save_labels: Boolean indicating whether to save text or not
    """
    
    # GET parameters
    if request.method == 'GET':
        #all_args = request.args # TODO: get all parameters in one line
        source = request.args.get('source')
        save_labels = request.args.get('save_labels')

    
    # POST parameters
    if request.method == 'POST':
        json_data = request.get_json() #Get the POSTed json
        json_data = json.dumps(json_data) # API receive a dictionary, so I have to do this to convert to string
        dict_data = json.loads(json_data) # Convert json to dictionary 
        source = dict_data['source']
        save_labels = dict_data.get('save_labels', None)        
    
    return source, save_labels
