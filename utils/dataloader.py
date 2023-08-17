#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:14:43 2023

@author: henry
"""

import cv2
import os
import glob
from pathlib import Path
import logging

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


class LoadImages:
    """
    A dataloader for loading images from specified paths.
    
    Attributes:
    - img_size (int): The desired size to resize the images to. Default is 224.
    - files (list): List of image file paths.
    - nf (int): Number of image files.
    - mode (str): Mode of operation, set to 'image'.
    """
    
    def __init__(self, path, img_size=224):
        """
        Initializes the LoadImages dataloader with the given path and image size.
        
        Parameters:
        - path (str, list, tuple): Path(s) to the images or directories.
        - img_size (int): The desired size to resize the images to. Default is 224.
        """
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        self.img_size = img_size
        self.files = images 
        self.nf = len(images)  # number of files
        self.mode = 'image'
        logging.info(f"Loaded {self.nf} images.")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        
        # Read image
        self.count += 1
        im0 = cv2.imread(path)  # BGR
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '
        
        # Resize image
        im = cv2.resize(im0, (self.img_size, self.img_size))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im / 255

        return path, im, im0, s
    
    def __len__(self):
        return self.nf  # number of files

