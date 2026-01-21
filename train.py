import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
# from ultralytics import RTDETR
import os
import shutil
import random
import torch

if __name__ == '__main__':
    model.train(data='data.yaml', #your dataset yaml
                cache=False,
                imgsz=640,
                # patience=0,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='YOLO-DB',
                )
