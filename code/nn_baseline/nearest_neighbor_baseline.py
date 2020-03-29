from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import random
import json
import sys
import argparse
import copy

import logging
import numpy

from PIL import Image, ImageFilter, ImageDraw

image_diff_dir = "data/resized_images/"

train_ann = {'annotations':[], 'images':[]  }

def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('-output_type',dest='output_type',default='single')  # single or multi
    args=parser.parse_args()
    return args

def PIL2array(img):
    return numpy.array(img.getdata(),numpy.uint8).reshape(img.size[1], img.size[0], 3)

def getFeats(image_id):
    im = Image.open( image_diff_dir + image_id + "_diff.jpg" )
    feats = PIL2array(im)/255.0
    feats = feats[:][:][0]
    feats = feats.reshape(-1)
    return feats

def getResnetFeats(image_id):
    feats1 = pickle.load( open('fc_feats/'+image_id+'.pickle', 'r') ).reshape(-1)
    feats2 = pickle.load( open('fc_feats/'+image_id+'_2.pickle', 'r') ).reshape(-1)
    print feats1.shape, feats2.shape
    feats = np.hstack([feats1, feats2])
    print feats.shape
    return feats.reshape(-1)

def getFeatsAndTexts(src, is_train=False, use_resnet_feats=False):
    data = json.load(open(src, 'r'))
    all_texts = []
    all_feats = []
    all_image = []
    for img_data in data:
        #print "img_data = ", img_data
        image_id = img_data['img_id'] # '9963' #
        #try:
        if use_resnet_feats:
            feats = getResnetFeats(image_id)
        else:
            feats = getFeats(image_id)
        #except:
        #    continue
        print "image_id = ", image_id
        all_texts.append(img_data['sentences'])
        all_feats.append(feats)
        all_image.append(image_id)
    print len(all_image), len(all_texts), len(all_feats)
    return np.array(all_feats), all_texts, all_image


####
# load data json
# go through image list
# load image features
# simultaneously load the texts as well
# for train and val/test


def main():

    params = parseArguments()
    train_json = "data/annotations/train.json"
    val_json = "data/annotations/val.json" # test.json
    use_resnet_feats=True

    train_feats, train_texts, train_image = getFeatsAndTexts(train_json,use_resnet_feats=use_resnet_feats)
    val_feats, val_texts, val_image = getFeatsAndTexts(val_json,use_resnet_feats=use_resnet_feats)
    print "train_feats.shape = ", train_feats.shape
    print "val_feats.shape = ", val_feats.shape
    num_feats = train_feats.shape[1]

    neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='auto').fit(train_feats)
    preds = { 'annotations':[], 'images':[]  }
    
    for i,(feats, text) in enumerate( zip(val_feats, val_texts) ):
    
        test_points = val_feats[i:i+1]
        distances, indices = nbrs.kneighbors(test_points)
        #print distances, indices
        output = train_texts[indices[0][0]]
        
        if output_type=="single":
            output = output[random.randint(0,len(output)-1)] # pick one; #randint(x,y): both x and y are inclusive
        else:
            output = ' '.join(output)
        print output.strip()
        tmp = {'caption':output.strip(), 'image_id':val_image[i] }
        preds['annotations'].append(tmp)
    
    json.dump(preds,open('predictions_nn_resnet_'+output_type+'.json','w') )


main()




