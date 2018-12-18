
from scipy.spatial import distance as dist
from collections import OrderedDict	
import numpy as np
from scipy.stats import itemfreq
import cv2
import math
import warnings
warnings.filterwarnings("ignore")

#Function to get the centroid of the Object.
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


#function to detect vehical/moving object 
def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=35):

    matches = []
    frame_copy=fg_mask
    # finding external contours
    im, contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (
            h >= min_contour_height)

        if not contour_valid:
            continue
        
        # getting center of the bounding box
        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches


#function to normalize the image so that the entire blob has the same rgb value
def normalized(down):
		s=down.shape
		x=s[1]
		y=s[0]
		norm=np.zeros((y,x,3),np.float32)
		norm_rgb=np.zeros((y,x,3),np.uint8)

		b=down[:,:,0]
		g=down[:,:,1]
		r=down[:,:,2]

		sum=b+g+r

		norm[:,:,0]=b/sum*255.0
		norm[:,:,1]=g/sum*255.0
		norm[:,:,2]=r/sum*255.0

		norm_rgb=cv2.convertScaleAbs(norm)
		return norm_rgb	
	

	




# initializing color class
colors = OrderedDict({"red": (255, 0, 0),"green": (0, 255, 0),"blue": (0,0, 255),"white":(255,255,255),"black":(100,100,100)})
lab = np.zeros((len(colors), 1, 3), dtype="uint8")
colorNames = []


f=open("output.txt","w")

incre=1
'''
if(len(x)==0):
	#no image name present in the file
	incre=1
else:
	#reding the image number 
	incre=int(x[-1].split(",")[0].split("_")[-1].split(".")[0])
f.close()
'''
#converting the rbg color to lab colors
for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
			lab[i] = rgb
			colorNames.append(name)
lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)


#function to label car lab color to a perticular color class
def label(image,lab,colorNames):

		# initialize the minimum distance found thus far
		minDist = (np.inf, None)
 
		# loop over the known L*a*b* color values
		for (i, row) in enumerate(lab):
			# compute the distance between the current L*a*b*
			# color value and the mean of the image
			
			d = dist.euclidean(row[0],image)
 
			# if the distance is smaller than the current distance,
			# then update the bookkeeping variable
			if d < minDist[0]:
				minDist = (d, i)
 
		# return the name of the color with the smallest distance
		return colorNames[minDist[1]]



#initialising background object used for background elemination 
background=cv2.createBackgroundSubtractorMOG2()


cap=cv2.VideoCapture('video_2.mp4')
#initialising frame counter
count_frame=0
while(cap.isOpened()):
	_,frame=cap.read()
	#resizing the frame 
	try:
		frame=cv2.resize(frame,(640,480))
	except:
		break
	#creating a copy of the frame
	frame_copy=frame
	frame_copy_copy=copy =frame[:,:]
	
	#applying background elemination
	bg=background.apply(frame)
	
	#additional image processing
	
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	bg= cv2.erode(bg,kernel,iterations = 1)
	
	# Fill any small holes
	closing=cv2.morphologyEx(bg,cv2.MORPH_CLOSE,kernel)
	cv2.imshow("closing",closing)
	
	# Remove noise
	opening=cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	cv2.imshow("removing_noise",opening)
	
	# Dilate to merge adjacent blobs
	dilation=cv2.dilate(opening, kernel, iterations=2)

	# threshold to remove furthur noise 
	dilation[dilation < 240] = 0
	bg=dilation
	
	#initialising output color list
	output_color=[]
	
	#detecting contour and calculating the co-ordinates of the contours
	contour_list=detect_vehicles(bg)
	
	#traversing through each detected contour 
	for ele in contour_list:
		x1=ele[0][0]
		y1=ele[0][1]
		x2=x1+ele[0][2]
		y2=y1+ele[0][3]
		#extracting the regions that contains car features
		
		slice_bg=frame_copy[y1:y2,x1:x2]
		
		#normalising the image so that there is uniform color throughout
		slice_bg=normalized(slice_bg)
		
		arr=np.float32(slice_bg)
		#reshaping the image to a linear form with 3-channels
		pixels=arr.reshape((-1,3))
		
		#number of clusters
		n_colors=2
		
		#number of iterations
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		
		#initialising centroid
		flags = cv2.KMEANS_RANDOM_CENTERS
		
		#applying k-means to detect prominant color in the image
		_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
		
		
		palette = np.uint8(centroids)
		quantized = palette[labels.flatten()]
		
		#detecting the centroid with densest cluster  
		dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
		


		r=int(dominant_color[0])
		g=int(dominant_color[1])
		b=int(dominant_color[2])

		
		rgb=np.zeros((1,1,3),dtype='uint8')
		rgb[0]=(r,g,b)
		
		
		
		#getting the  label of the car color
		color=label(rgb,lab,colorNames)
		
		
		output_color.append(color)
		
		#drawing rectangle over the detected car 
		frame_copy= cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(r,g,b),3)
		font = cv2.FONT_HERSHEY_SIMPLEX
		#labeling each rectangle with the detected color of the car
		cv2.putText(frame_copy,color,(x1,y1), font, 2,(r,g,b),2,cv2.LINE_AA)
	#openinig file to write the ouput of each frame
	#f=open("output.txt","w")
	
	#writing onto the file for every 10 frames
	
	if(count_frame%10==0):
		if(len(output_color)!=0):
			c=",".join(output_color)+'\n'
			
			#image_name="img_"+str(incre)+".jpg,"+c+'\n'
			f.write(c)
			# cv2.imwrite(img,frame)
			incre=incre+1
			count_frame=0
	count_frame+=1
	cv2.imshow("object",frame_copy)
	if(cv2.waitKey(30)==27 & 0xff):
		break

cap.release()
cv2.destroyAllWindows()