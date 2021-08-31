from yolo import YOLO
import logging as log
import cv2
import numpy as np
import os
import torch


log.basicConfig(level=log.DEBUG)



def init():
    """Initialize model

    Returns: model

    """
    model_path="/usr/local/ev_sdk/model/Epoch40-Total_Loss5.1861-Val_Loss4.8182.pth"
    
    if not os.path.isfile(model_path):
        log.error(f'{model_path} does not exist')
        return None
    log.info('Loading model...')
    #state_dict = torch.jit.load(model_path)

    yolo_model = YOLO(model_path)
    
    return yolo_model


def process_image(net, input_image, args=None):

    
    return  net.detect_image(input_image)



if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/home/data/52/4.jpg')

    predictor  = init()
    
    result = process_image(predictor, img)

    log.info(result)


