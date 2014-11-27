# Importing libraries
import cv2
import cv
from math import *
from numpy import *
#from sympy import Symbol,cos,sin
from operator import *
from numpy.linalg import *
import time
import ctypes
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
# Prints the numbers in float instead of scientific format
set_printoptions(suppress=True)

filename='UCI HAR Dataset/train/' # Dataset Used
#----------------------------------------------------------------------------------------------#
#This function reads the manual correspondences saved in a text file.
def readmatches(filename):    
    f = open(filename).read()
    rows = []
    for line in f.split('\n'):
        rows.append(line.split(' '))
    rows.pop()
    #for loopVar1 in range(0, len(rows)):
    #	for loopVar2 in range(0, len(rows[loopVar1])):
    #		rows[loopVar1][loopVar2]=float(rows[loopVar1][loopVar2])
    return rows 

#--------------------------------------------------------------------------------------------#
#This function saves the homographies to a text file 
def save_matrix(filename, H):
	fo = open(filename, 'w', 0)
	for loopVar1 in range(H.shape[0]):
		for loopVar2 in range(H.shape[1]):
			fo.write(str(H[loopVar1, loopVar2]))
			if loopVar2!=H.shape[1]-1:		
				fo.write('\t')
		if loopVar1!=H.shape[0]-1:
			fo.write('\n')
	fo.close()	
		
#---------------------------------------------------------------------------------------------#
# This function takes in P1, P2, F and epipoles to find H1 and H2
def Rectify_Images(correspondences, F, P1, P2, e1, e2, H1, H2):
	img_1 = cv2.imread(filename+'Pic_1.jpg',1) # Read two images
	img_2 = cv2.imread(filename+'Pic_2.jpg',1)
	image_width=img_1.shape[1]			# Get their size
	image_height=img_1.shape[0]

	G=matrix(identity(3))				# Initialize required matrices
	R=matrix(zeros((3,3)))
	T=matrix(identity(3))
	H2=matrix(zeros((3,3)))

	T[0,2]=(-image_width/2)				# Calculate T
	T[1,2]=(-image_height/2)
	
	e2=T*e2						# Find translated epipole
	mirror = e2[0,0] < 0				

	d=sqrt(pow((e2[1,0]),2)+pow((e2[0,0]),2))	# Find distance between origin and epipole
	alpha=e2[0,0]/d					# Cos theta 
	beta=e2[1,0]/d					# -Sin theta

	R[0,0]=alpha#cos(theta)				# Calculate R
	R[0,1]=beta#-sin(theta)
	R[1,0]=-beta#sin(theta)
	R[1,1]=alpha#cos(theta)
	R[2,2]=1

	e2=R*e2						# Find rotated epipole

	f=e2[0,0]/e2[2,0]				# Calculate G
	G[2,0]=(-1/f)	
	print f

	H2=G*R*T					# Calculate H2
	print 'H2', H2
	print 'e2 after H2', G*e2			# Check Epipole after sending to infinity
	print H2*transpose(matrix([image_width/2, image_height/2, 1]))
	
	center_point = matrix(zeros((3,1)))		# Calculate T2
	center_point[0,0]=image_width/2
	center_point[1,0]=image_height/2
	center_point[2,0]=1
	new_center=H2*center_point
	
	#print center_point, new_center
	T2=matrix(identity(3))
	T2[0,2]=(image_width/2)-(new_center[0,0]/new_center[2,0])
	T2[1,2]=(image_height/2)-(new_center[1,0]/new_center[2,0])
	#print T2
	H2=T2*H2					# Calculate H2 after correcting for T2
	
	if mirror:					# Mirror H2 if epipole is along negative x-axis
        	mm = array([[-1, 0, image_width],
                       [0, -1, image_height],
                       [0, 0, 1]], dtype=float)
       		#H1 = mm.dot(H1)
       		H2 = mm.dot(H2)

	print 'H2', H2
	print 'center after H2', H2*transpose(matrix([image_width/2, image_height/2, 1]))
	#-------------------------------------------------------------------------------------#
	H_temp1=H2*P2*pinv(P1)				# Find Temporary H1
	A=[]
	b=[]
	for loopVar1 in range(len(correspondences)):	# For all correspondences, find A and b for least squares estimation of H0
		x1=[correspondences[loopVar1, 0], correspondences[loopVar1, 1], 1]
		new_x1=H_temp1*transpose(matrix(x1))
		new_x1=asarray(new_x1/new_x1[2,0])
		x2=[correspondences[loopVar1, 2], correspondences[loopVar1, 3], 1]
		new_x2=H2*transpose(matrix(x2))
		new_x2=asarray(new_x2/new_x2[2,0])
		#print new_x1, new_x2
		A.append([new_x1[0][0], new_x1[1][0], new_x1[2][0]])
		b.append(new_x2[0][0])
	h=linalg.lstsq(A, b)[0]				# Calculate H0
	#print h
	H_temp2=matrix(identity(3))
	H_temp2[0,0]=h[0]
	H_temp2[0,1]=h[1]
	H_temp2[0,2]=h[2]

	#print H_temp2
	H1=H_temp2*H_temp1				# Calculate H1
	#print H1	
		
	center_point = matrix(zeros((3,1)))		# Calculate T2
	center_point[0,0]=image_width/2
	center_point[1,0]=image_height/2
	center_point[2,0]=1
	new_center=H1*center_point
	
	T2=matrix(identity(3))
	T2[0,2]=(image_width/2)-(new_center[0,0]/new_center[2,0])
	T2[1,2]=(image_height/2)-(new_center[1,0]/new_center[2,0])
	#print T2	
	H1=T2*H1					# Calculate H1 after correction for T2
	
	if mirror:					# Mirror H1 if epipole is along negative x-axis
        	mm = array([[-1, 0, image_width],
                       [0, -1, image_height],
                       [0, 0, 1]], dtype=float)
       		#H1 = mm.dot(H1)
       		#H2 = mm.dot(H2)	
	print 'H1', H1

	cv2.imwrite(filename+'Pic_2_corrected.jpg', Apply_Homography(img_2, H2)) #Save the rectified images
	cv2.imwrite(filename+'Pic_1_corrected.jpg', Apply_Homography(img_1, H1))	
	return 0

#---------------------------------------------------------------------------------------------#
# This function takes in an image and homography matrix and applies the given homography to produce new image
def Apply_Homography(image, H):	
	#Declare two empty arrays for storing points while applying homography
	old_point=[]
	new_point=[]
	image_width=image.shape[1]	#width
	image_height=image.shape[0]	#height
	inv_H=inv(H)
	print inv_H, 'inv_H'
	
	#---------------------------------------------------------------------------------------------#
	#This section of code finds out minimum and maximum indices in both x and y. 
	#Later, finds out the scaling factor to scale the final output image
	#Creates an empty output image with scaled-down dimensions
	min_x=0
	min_y=0
	max_x=0
	max_y=0

	a=image.shape[0]	#height
	b=image.shape[1]	#width
	corners=[[0, 0],[b, 0],[0, a],[b, a]]
	for loopVar1 in range(0,len(corners)):
		old_point=[[corners[loopVar1][0]],[corners[loopVar1][1]],[1]]
		print old_point
		new_point=H*old_point
		new_point=new_point*(1/new_point[2][0])
		old_point=array(old_point)
		new_point=array(new_point)
		if (loopVar1==0):
			min_x=new_point[0][0]
			min_y=new_point[1][0]
			max_x=new_point[0][0]
			max_y=new_point[1][0]	
		else:
			if(new_point[0][0]<min_x):
				min_x=new_point[0][0]
			if(new_point[1][0]<min_y):
				min_y=new_point[1][0]
			if(new_point[0][0]>max_x):
				max_x=new_point[0][0]
			if(new_point[1][0]>max_y):
				max_y=new_point[1][0]
		print loopVar1

	print min_x
	print min_y
	print max_x
	print max_y

	min_x=0
	min_y=0
	b=image.shape[1]
	a=image.shape[0]
	scaling_x=1
	output_img = zeros((a,b,3), uint8) # Output Image with all pixels set to black
	#---------------------------------------------------------------------------------------------#

	#---------------------------------------------------------------------------------------------#
	#This loop applies inverse homography to all points in output image and gets original pixel coordinates.
	#Used Inverse transformation to avoid having empty pixels in the output image
	for loopVar1 in range(0,a):
		for loopVar2 in range(0,b):
			new_point=matrix([[(loopVar2*scaling_x)+min_x],[(loopVar1*scaling_x)+min_y],[1]])
			old_point=inv_H*new_point
			old_point=old_point*(1/old_point[2][0])
			old_point=array(old_point)
			new_point=array(new_point)
		
		#When indices are positive, copy from original image.
			if ((old_point[0][0]>0)and(old_point[1][0]>0)): 
				try:
					output_img[loopVar1][loopVar2]=image[old_point[1][0]][old_point[0][0]]	
				#When indices exceed the available image size,keep the black pixel as it is in the output image.	
				except IndexError:
					output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
			#When indices are negative, keep the black pixel as it is in the output image.		
			else:
				output_img[loopVar1][loopVar2]=output_img[loopVar1][loopVar2]
		print loopVar1			
	return output_img
#----------------------------------------------------------------------------------------------#
# This function is the cost function used by Lev-Mar optimization. It takes in a parameter vector and returs the current cost vector
def CostFunction(p):
	P2=matrix(array(p).reshape((3,4)))		# Find P2 from parameter p
	est_x=[]				
	for loopVar1 in range(len(op)):			# For all correspondences, do triangulation
		A=matrix(zeros((4,4)))			# Find Matrix A
		A[0,:]=(op[loopVar1, 0]*P1[2,:])-(P1[0,:])
		A[1,:]=(op[loopVar1, 1]*P1[2,:])-(P1[1,:])
		A[2,:]=(op[loopVar1, 2]*P2[2,:])-(P2[0,:])
		A[3,:]=(op[loopVar1, 3]*P2[2,:])-(P2[1,:])

		world_X=transpose(matrix((linalg.svd(transpose(matrix(A))*matrix(A))[2][3]).tolist()[0])) # Find world point
		
		proj_x=P1*world_X				# Project the world point back to Image 1 and Image 2
		proj_x=proj_x/proj_x[2,0]
		proj_x_bar=P2*world_X
		proj_x_bar=proj_x_bar/proj_x_bar[2,0]
		
		est_x.append(proj_x[0,0])			# Take the projected points as estimated
		est_x.append(proj_x[1,0])
		est_x.append(proj_x_bar[0,0])
		est_x.append(proj_x_bar[1,0])
		
	cost=subtract(X,est_x)					# Find the cost by subtrating estimates from known image points
	return cost						# Return the cost vector

#----------------------------------------------------------------------------------------------#
# This function normalizes the selected points so that the rectification works for all scales
def normalize(correspondences):
	mean_x_1 = 0.0
	mean_y_1 = 0.0
	mean_x_2 = 0.0
	mean_y_2 = 0.0

	for loopVar1 in range(NUM_OF_POINTS):			# For all correspondences, find means
		mean_x_1+=correspondences[loopVar1, 0]
		mean_y_1+=correspondences[loopVar1, 1]
		mean_x_2+=correspondences[loopVar1, 2]
		mean_y_2+=correspondences[loopVar1, 3]
		
	mean_x_1/=float(NUM_OF_POINTS)			
	mean_y_1/=float(NUM_OF_POINTS)
	mean_x_2/=float(NUM_OF_POINTS)
	mean_y_2/=float(NUM_OF_POINTS)
	
	variance_1 = 0.0
	variance_2 = 0.0
	for loopVar1 in range(NUM_OF_POINTS):			# For all correspondences, find variances
		variance_1+=sqrt((correspondences[loopVar1, 0] - mean_x_1)**2 + (correspondences[loopVar1, 1] - mean_y_1)**2)
		variance_2+=sqrt((correspondences[loopVar1, 2] - mean_x_2)**2 + (correspondences[loopVar1, 3] - mean_y_2)**2)
	variance_1/=float(NUM_OF_POINTS)
	variance_2/=float(NUM_OF_POINTS)

	scale_1 = sqrt(2)/variance_1			# Find Scales
	scale_2 = sqrt(2)/variance_2

	translate_x_1 = -scale_1*mean_x_1		# Find translation factors
	translate_y_1 = -scale_1*mean_y_1

	translate_x_2 = -scale_2*mean_x_2
	translate_y_2 = -scale_2*mean_y_2

	T1 = matrix(zeros((3,3)))			# Initialize T1 and T2
	T2 = matrix(zeros((3,3)))

	T1[0, 0]= scale_1				# Calculate T1
	T1[0, 2]= translate_x_1
	T1[1, 2]= translate_y_1
	T1[1, 1]= scale_1
	T1[2, 2]= 1
	
	T2[0, 0]= scale_2				# Calculate T2
	T2[0, 2]= translate_x_2
	T2[1, 2]= translate_y_2
	T2[1, 1]= scale_2
	T2[2, 2]= 1

	return T1, T2					# Return the T1 and T2 matrices

#----------------------------------------------------------------------------------------------#


# Main Code starts
op=matrix(readmatches(filename+'X_train.txt'))		# Read the manual correspondences from the file
NUM_OF_POINTS=op.shape[0]						# Find the number of points
print NUM_OF_POINTS, op.shape[1]
