from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import math
import sys

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt



#############
def changeToBinary(pixels_matrix,rgb):
	shap = pixels_matrix.shape
	ret = np.zeros((shap[0],shap[1],1))
	for i,row in enumerate(pixels_matrix):
		for j,val in enumerate(row):
			if rgb:
				tmp = sum(val) / (1.0 * len(val))
			else:
				tmp = val
			if round(tmp) > 127:
				ret[i][j] = [0]
			else:
				ret[i][j] = [1]
	return ret


def PIL2array(img):
	vals = np.array(img.getdata(),np.uint8)
	num_channels = 0
	try:
		num_channels = vals.shape[1]
	except:
		num_channels = 1
	return vals.reshape(img.size[1], img.size[0], num_channels), num_channels


def getMatrixFromIM(im,rgb):
	ret, _ = PIL2array(im)
	return ret


def loadImageAndProcess(image_name, change_to_binary=True, transpose = True, rgb=True):
	print("image_name,rgb = ",image_name, rgb)
	im = Image.open( image_name )
	pixels = getMatrixFromIM(im,rgb=rgb)
	#print "pixles.shape before binarizing = ", pixels.shape
	if change_to_binary:
		pixels = changeToBinary(pixels,rgb=rgb)
		if transpose:
			pixels = np.transpose(pixels,[1,0,2])
	return pixels


###############################


def getPixelCoordinates(pixels, pixel_value=1, flip_y=False): # returns list of pixel coordinates with value = pixel_value
	# assuming 2 dimensional pixels
	shap = pixels.shape
	assert len(shap) == 2 or (len(shap)==3 and len(pixels[0][0])==1)
	third_dim = False
	if len(shap)==3 and len(pixels[0][0])==1:
		third_dim = True
	#print("third_dim = ",third_dim)
	#print("flip_y = ",flip_y)
	ret = []
	for i in range(shap[0]-1,0,-1):
		for j in range(shap[1]):
			val = pixels[i][j]
			if third_dim:
				val = val[0]
			if val == pixel_value:
				if flip_y:
					ret.append([i,shap[1]-1-j])
				else:
					ret.append([i,j])
	return np.array(ret)


###############################

def drawRectanglesOnImage(image_path, rectangles_coordinates, save_path, show_image, classes=None):
	im = Image.open(image_path)
	fill_color = 128
	fill_color_maps = {0:5,1:128,-1:255}
	#pixels = loadImageAndProcess(image_path,change_to_binary=False,rgb=True)
	#im = getImgFromPixels( pixels, rgb=True )
	image_y_limit = im.size[1]
	draw = ImageDraw.Draw(im)
	for i,rectangle  in enumerate(rectangles_coordinates):
		x1,x2,y1,y2 = rectangle
		x1,x2,y1,y2 = x1,x2,image_y_limit - y1, image_y_limit - y2
		if classes is not None:
			fill_color = fill_color_maps[classes[i]]
		draw.line((x1,y1) + (x1,y2), fill=fill_color)
		draw.line((x2,y1) + (x2,y2), fill=fill_color)
		draw.line((x1,y1) + (x2,y1), fill=fill_color)
		draw.line((x1,y2) + (x2,y2), fill=fill_color)
	del draw
	if save_path!=None:
		print("Saving to ", save_path)
		im.save(save_path, "JPEG", quality=100, optimize=False, progressive=True)
	if show_image:
		im.show()



############################### Testing the library
def test(img_name = 'images(35).png'):
	im = Image.open( img_name )
	print(im.size)
	im_width, im_height = im.size
	im_pixels = list(im.getdata())
	print(len(im_pixels), len(im_pixels[0])) # im_width*im_height, 4

	# change to binary
	all_pixels = []
	for cpixel in im_pixels:
		#if round(sum(cpixel)) / float(len(cpixel)) > 127:
		if round(sum(cpixel[:3]) / 3.0) > 127:
			all_pixels.append(0)
		else:
			all_pixels.append(1)
	print(sum(all_pixels))


