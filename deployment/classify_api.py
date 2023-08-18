#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:22:22 2023

@author: henry

Run classify inference on images
Usage - sources:
    $ python classify_api.py

"""

import logging
import argparse
import os
import sys
import cv2
import json
import wandb
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, Response, request
from pathlib import Path
from waitress import serve

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import check_file, select_device, time_synchronized, increment_path, annotate, update_options
from utils.dataloader import IMG_FORMATS, LoadImages
from utils.model import load_model

# Setup logging
script_name = Path(__file__).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{script_name}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


# Initialize flask API
app = Flask(__name__)

def run(opt):
    """
    Perform model inference on a set of images.

    Parameters:
    - source (str or Path): Path to the image source.
    - labels (str or Path): Path to the labels.
    - imgsz (int): Image size for inference.
    - view_img (bool): Display the results.
    - save_labels (bool): Save results to text files.
    - nosave (bool): Do not save images.
    - project (str or Path): Path to save results.
    - name (str): Name of the save directory.
    - half (bool): Use FP16 half-precision inference.

    Returns:
    None. Results are saved to the specified directory or displayed.
    """
    
    source=opt.source
    labels=opt.labels
    imgsz=opt.imgsz
    view_img=opt.view_img
    save_labels=opt.save_labels
    nosave=opt.nosave
    project=opt.project
    name=opt.name
    half=opt.half
    
    # Check source
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS)
    is_url = source.lower().startswith(('http://', 'https://'))
        
    # Directories
    save_dir = increment_path(Path(project) / name)  # increment run
    (save_dir / 'labels' if save_labels else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (ROOT / 'raw_images').mkdir(parents=True, exist_ok=True)  # make dir raw_images
    
    if is_url and is_file:
        source = check_file(source, raw_path = ROOT / 'raw_images' )  # download
    
    # Download labels file in case it is a url
    labels = str(labels)
    if labels.lower().startswith(('http://', 'https://')):
        labels = check_file(labels)

    # Load model and labels
    labels = pd.read_csv(labels).name.to_numpy()
    
    # Model is loaded outside of this function
    # with tf.device(device):
    #     model = load_model(weights)
        
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    _ = model(tf.expand_dims(tf.random.normal((imgsz,imgsz,3)),0))  # warmup
    init_time = time_synchronized()
    
    for path, im, im0, s in dataset:
        t0 = time_synchronized()
        with tf.device(device):
            image_tensor = tf.convert_to_tensor(im, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor,0)
        
        # Inference
        with tf.device(device):
            pred = model(image_tensor)['default']
        t1 = time_synchronized()

        # Process predictions. It is a loop, because in the future is expected to evaluate several batch sizes
        for i, prob in enumerate(pred):  # per image

            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) #+ ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt

            # Top results
            top3i = prob.numpy().argsort()[::-1][0:3] # top 3 indices

            # Write results
            text = '\n'.join(f'{prob[j]:.2f},{labels[j]}' for j in top3i)
            if save_img or view_img:  # Add bbox to image
                annotate(im0, text, txt_color=(255, 255, 255))
            if save_labels:  # Write to file
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(text + '\n')

            # Show results
            if view_img:
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image': # it is expected to evaluate videos in the future
                    cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        logging.info(f'{s}{prob[top3i[0]]:.2f} {labels[top3i[0]]} {(t1-t0) * 1E3:.1f}ms') #filename + prob + class + time


        # This is done in order to be shown in a browser, save_labels will return json file, otherwise, an image in bytes
        if save_labels:
            if os.path.exists(txt_path + '.txt'):
                result = pd.read_csv(txt_path + '.txt', sep =",",names = ['conf', 'class'], header = None)
                result = result.to_json(orient='records')
                result = json.loads(result)
                
            else:
                result = []
            dict_result=dict()
            dict_result["results"]=result
            yield json.dumps(dict_result)
        else:
            im0 = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')
            

    # Print results
    logging.info(f'Done. The entire process took ({time_synchronized() - init_time:.3f}s)')
    if save_labels or save_img:
        logging.info(f"Results saved to {save_dir}")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    
    # Get request parameters and update opt
    # If request doesn't contain files, update your options and continue
    if not request.files.getlist('myfile'):
        opt.source, opt.save_labels = update_options(request)
    else:
        uploaded_file = request.files['myfile']
        source = 'test_public_123'+str(uploaded_file.filename)
        uploaded_file.save(source)
        opt.save_labels = request.form.get('save_labels')
        opt.source = source
    
    
        
    return Response(run(opt), mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= 'hdnh2006/bird_classifier/model_w5cxp2z2:v0', help='model path of wandb or local path that contains .pb file')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--labels', type=str, default=ROOT / 'data/aiy_birds_V1_labelmap.csv', help='(optional) dataset.csv path or url')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size h,w in case the model accept dynamic batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-labels', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images')
    parser.add_argument('--project', default=ROOT / 'runs/predict', help='save results to project/name')
    parser.add_argument('--name', default='inference', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    
    session = wandb.init('classifier-deployment')
    
    # Load model
    device = select_device(opt.device)
    with tf.device(device):
        model = load_model(session, opt.weights)
    
    # Run app
    # Testing purposes
    # app.run(host="0.0.0.0", port=opt.port, debug=False) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)

    # Production, avoid warning development env in Flask
    serve(app, host="0.0.0.0", port=opt.port) 

    session.finish()
