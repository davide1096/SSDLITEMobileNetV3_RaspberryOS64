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
parser.add_argument('-i', '--input', default='camera', 
                    help='path to input input image')
parser.add_argument('-t', '--threshold', default=0.5, type=float,
                    help='detection threshold')
args = vars(parser.parse_args())

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)

#timing
t = time.perf_counter()
# read the image
if args['input']=='camera':
    os.system('libcamera-jpeg -o still.jpg -n')
    image = Image.open('still.jpg')
else:
    image = Image.open(args['input'])
#image = image.resize((320,320), Image.ANTIALIAS)
# detect outputs
inf_t = time.perf_counter()
boxes, classes, labels = detect_utils.predict(image, model, device, args['threshold'])
# draw bounding boxes
image = detect_utils.draw_boxes(boxes, classes, labels, image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{''.join(str(args['threshold']).split('.'))}"
# cv2.imshow('Image', image)
cv2.imwrite(f"output/{save_name}.jpg", image)
total_t = time.perf_counter()-t
total_inf_t = time.perf_counter()-inf_t
# cv2.waitKey(0)
print("Finished in {}, with inference needing {}".format(total_t, total_inf_t))
