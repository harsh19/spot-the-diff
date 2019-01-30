import numpy as np
import math
from PIL import Image, ImageFilter
#import matplotlib.pyplot as plt
import pickle


import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utilities



def loadClusters(labels,X,core_samples_mask,shap, ignore_noise):
	cluster_coords = {}
	for label,xy in zip(labels,X):
		if ignore_noise:
			if label==-1: # label for noise
				continue
		if label not in cluster_coords:
			cluster_coords[label] = []
		cluster_coords[label].append(xy)
	#print len(cluster_coords)
	#print cluster_coords.items()[0][1][0][0]
	return cluster_coords

def findRectangleCoordinates(cluster_coords):
	vals = []
	for cluster_label, points in cluster_coords.items():
		points = np.array(points)
		tmp = [ np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1]) ] # minx,maxx,miny,maxy
		vals.append(tmp)
	return vals

def findBoxPoints(cluster_plot_path_prefix, ignore_noise=False):
	vals = pickle.load( open(cluster_plot_path_prefix+"_vals.obj","rb") )
	labels,X,core_samples_mask,shap = vals
	# cluster_label -> set of points. Each point is xy tuple
	cluster_coords = loadClusters(labels,X,core_samples_mask,shap, ignore_noise) 
	rectangles_coordinates = findRectangleCoordinates(cluster_coords)
	return rectangles_coordinates

def findAdjustedRectangleCoordinates(rectangles_coordinates, alignment):
	# rectangle_coordinates: each items is [minx,maxx,miny,maxy]
	minw,minh,min_loss = alignment
	print("w,h = ", minw, minh)
	for i in range(len(rectangles_coordinates)):
		rectangles_coordinates[i][0]-=minw
		rectangles_coordinates[i][1]-=minw
		rectangles_coordinates[i][2]+=minh
		rectangles_coordinates[i][3]+=minh
	return rectangles_coordinates


if __name__ == "__main__":
	data_path = "../../Data/samples/" #"/mnt/tir2/"
	cluster_data_path = data_path + "all_images_cluster/"
	images_path = "" # "resized_images/"
	save_path_prefix = data_path + "boxed_images/"
	
	img = "235"
	rectangles_coordinates = findBoxPoints(cluster_data_path+img)
	utilities.drawRectanglesOnImage(image_path=data_path+img+".png", rectangles_coordinates=rectangles_coordinates, save_path=save_path_prefix+img+".jpg", show_image=False)
	utilities.drawRectanglesOnImage(image_path=data_path+img+"_2.png", rectangles_coordinates=rectangles_coordinates, save_path=save_path_prefix+img+"_2.jpg", show_image=False)
	
