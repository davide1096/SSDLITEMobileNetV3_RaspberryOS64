import torch
import argparse
import cv2
import detect_utils
import os
import time

from PIL import Image
from model import get_model

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--timelimit', default=0, type=int, 
                    help='time limit')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')
args = vars(parser.parse_args())

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)

#initializations
start_t = time.perf_counter()
first = True
#loop
while True:
    os.system('libcamera-jpeg -o still.jpg -n')
    elapsed_time = time.perf_counter()-start_t
    if (args['timelimit']!=0) & (elapsed_time>float(args['timelimit'])):
        print("Reached the time limit!")
        break

    image = Image.open('still.jpg')
    boxes, classes, labels = detect_utils.predict(image, model, device, args['threshold'])
    # draw bounding boxes
    image = detect_utils.draw_boxes(boxes, classes, labels, image)
    #if first:
    #   os.mkdir('/output/dummydir')
    #    first = False
    save_name = str(elapsed_time)[0:3]
    cv2.imwrite(f"output/dummy/{save_name}.jpg", image)
