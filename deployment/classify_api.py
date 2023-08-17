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
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, Response, request
from pathlib import Path

from utils.general import check_file, select_device, time_synchronized, increment_path, annotate
from utils.dataloader import IMG_FORMATS, LoadImages
from utils.model import load_model

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv8API root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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

def detect(opt):
    weights=opt.weights  # model path or triton URL
    source=opt.source  # file/dir/URL/glob/screen/0(webcam)
    imgsz=opt.imgsz  # inference size (height, width)
    conf_thres=opt.conf_thres  # confidence threshold
    iou_thres=opt.iou_thres  # NMS IOU threshold
    max_det=opt.max_det  # maximum detections per image
    view_img=opt.view_img  # show results
    save_txt=opt.save_txt  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=opt.save_crop  # save cropped prediction boxes
    nosave=opt.nosave  # do not save images/videos
    classes=opt.classes  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=opt.agnostic_nms  # class-agnostic NMS
    augment=opt.augment  # augmented inference
    visualize=opt.visualize  # visualize features
    update=opt.update  # update all models
    project=opt.project  # save results to project/name
    name=opt.name  # save results to project/name
    exist_ok=opt.exist_ok  # existing project/name ok, do not increment
    line_thickness=opt.line_thickness  # bounding box thickness (pixels)
    hide_labels=opt.hide_labels  # hide labels
    hide_conf=opt.hide_conf  # hide confidences
    vid_stride=opt.vid_stride  # video frame-rate stride

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model (outside of the function)
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
        # This is done in order to be shown in a browser, save_txt will return json file, otherwise, an image in bytes
        if save_txt:
            if os.path.exists(txt_path + '.txt'):
                result = pd.read_csv(txt_path + '.txt', sep =" ",names = ["class","x","y","w","h","conf"], header = None)
                result = result.to_json(orient="records")
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
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    
    # Get request parameters and update opt
    # If request doesn't contain files, update your options and continue
    if not request.files.getlist('myfile'):
        opt.source, opt.save_txt = update_options(request)
    else:
        uploaded_file = request.files['myfile']
        url = 'test_public_123'+str(uploaded_file.filename)
        uploaded_file.save(url)
        opt.save_txt = None
        opt.source = url
        
    return Response(detect(opt), mimetype='multipart/x-mixed-replace; boundary=frame')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= 'hdnh2006/bird_classifier/model_w5cxp2z2:v0', help='model path of wandb or local')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--labels', type=str, default=ROOT / 'data/aiy_birds_V1_labelmap.csv', help='(optional) dataset.csv path or url')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size h,w in case the model accept dynamic batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-labels', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save results to project/name')
    parser.add_argument('--name', default='inference', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')    
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    
    # Load model
    with tf.device(device):
        model = load_model(weights)
    opt.device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16 = opt.half)
    stride, names, pt = model.stride, model.names, model.pt
    
    #main(opt)

    # Run app
    app.run(host="0.0.0.0", port=opt.port, debug=False) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)