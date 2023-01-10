import numpy as np
import cv2 
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
import matplotlib.pyplot as plt
import pandas as pd
import ast


df_listing = pd.read_csv("data/listing.csv")


# get_pics
def get_pics(listing_id):
    directories = ast.literal_eval(df_listing[df_listing["listing_id"]==listing_id].pictures.values[0])
    pictures = []
    for d in directories:
        pictures.append(cv2.imread(d))
    print(len(pictures))
    return pictures

#visiualiser les photos 
def plot_imgs(listing_id):
    pictures = get_pics(listing_id)
    fig = plt.figure(figsize=(10, 10))
    for p in range(len(pictures)):
        plt.subplot(5,3,p+1)
        plt.imshow(pictures[p])
        plt.axis('off')
    plt.show()


## Compute sift matching score
def sift_score(image1, image2, ratio = 0.8) :

    hight = max(np.shape(image1)[0],np.shape(image2)[0])
    width = max(np.shape(image1)[1],np.shape(image2)[1])

    image1 = cv2.resize(np.array(image1), (width, hight))                           
    image2 = cv2.resize(np.array(image2), (width, hight))

    #sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    try : 
        matches = bf.knnMatch( descriptors_1, descriptors_2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append([m])
        number_keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            number_keypoints = len(keypoints_1)
        else : 
            number_keypoints = len(keypoints_2)

        score = len(good)/number_keypoints*100
    except:
        score = 0
        good, keypoints_1, keypoints_2, number_keypoints = None, None, None, None
        
    return score, keypoints_1, keypoints_2 ,good, number_keypoints


### Compute object detection

def object_detection(image_dir, thresh=0.7):
    
    image = Image.open(image_dir)
    
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
    
    
    objects, boxs, labels, scores = [], [], [], []
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()] 
        x,y,w,h = box
        if score > thresh:
            objects.append(image.crop((int(x), int(y), int(w), int(h))))
            boxs.append([int(x), int(y), int(w), int(h)])
            labels.append(model.config.id2label[label.item()])
            scores.append(round(score.item(), 3))
    return  objects, labels, scores, boxs

def plot_object(image_dir):
    image = cv2.imread(image_dir)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    objects, labels, scores, boxs = object_detection(image_dir,  thresh=0.7)
    
    for x,y,w,h in boxs : 
        image = cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)
        
    plt.figure(figsize=(15, 20))
    plt.imshow(image),plt.show()
