# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import time
import scipy.io
import shutil

def yolo_detect(img,pathOut = '../test1.jpg',
                label_path='../cfg/coco.names',
                config_path='../cfg/yolov3_coco.cfg',
                weights_path='../cfg/yolov3_coco.weights',
                confidence_thre=0.5,
                nms_thre=0.3,
                jpg_quality=80):

    '''
    pathIn：The path of the original image 
    pathOut：The path of the output image 
    label_path：The path of the category label file
    config_path：The path of the model configuration file
    weights_path：The path of the model weight file
    confidence_thre：0-1, the confidence (probability/scoring) threshold, that is, to retain the bounding box with a probability greater than this value, the default is 0.5
    nms_thre：threshold for non-maximum suppression, the default is 0.3
    jpg_quality：set the quality of the output image, the range is 0 to 100, the default is 80, the larger the quality, the better 

    '''


    LABELS = open(label_path).read().strip().split("\n")
    nclass = len(LABELS)
    
    # Randomly match the corresponding color for the bounding box of each category 
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    
    # Load the picture and get its dimensions 
    #base_path = os.path.basename(pathIn)
    #img = cv2.imread(pathIn)
    (H, W) = img.shape[:2]
    
    # Load model configuration and weight files
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Get the name of the YOLO output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Build the picture into a blob, set the picture size, and then execute it once
    # YOLO feedforward network calculation, and finally obtain the bounding box and corresponding probability 
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    

    # Initialize bounding box, confidence (probability) and category
    boxes = []
    confidences = []
    classIDs = []
    
    # Iterate each output layer, a total of three
    for output in layerOutputs:
        # Iterate each detcetation
        for detection in output:
            # Extract category ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # Only keep bounding boxes with confidence greater than a certain value
            if classID==0: ##Filter category 
                if confidence > confidence_thre:
                    # Restore the coordinates of the bounding box to match the original picture, remember that YOLO returns
                    # The center coordinates of the bounding box and the width and height of the bounding box 
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
    
                
                    # Calculate the position of the upper left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
                    # Update bounding box, confidence (probability) and category 
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    
    # Use non-maximum suppression methods to suppress weak, overlapping bounding boxes 
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    

    if len(idxs) > 0:

        shutil.rmtree('./roi_lib/gallery/img')
        os.mkdir('./roi_lib/gallery/img')
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if x<0: x=0
            if y<0: y=0

            ROI=img[y:y+h,x:x+w]
            cv2.imwrite('./roi_lib/gallery/img/%d.jpg'%(i), ROI)
        os.system("python tools/inference.py")
        with open('./roi_lib/test.txt', 'r') as f: 
            index_result,score_result = f.read().split(":")
            index_result=int(index_result)
            score_result=float(score_result)
        (x, y) = (boxes[index_result][0], boxes[index_result][1])
        (w, h) = (boxes[index_result][2], boxes[index_result][3])
    else:
        x,y,w,h,score_result=0,0,0,0,0

           
    return x,y,w,h,score_result
