import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
	
class Skin_Detect():
	def __init__(self):
		pass
	#RGB bounding rule
	def Rule_A(self,BGR_Frame,plot=False):
		'''this function implements the RGB bounding rule algorithm
		--inputs: 
		BGR_Frame: BGR components of an image
		plot: Bool type variable,if set to True draw the output of the algorithm
		--return a anumpy array of type bool
		'''
		B_Frame, G_Frame, R_Frame =  [BGR_Frame[...,BGR] for BGR in range(3)]
		#B_Frame, G_Frame, R_Frame  = cv2.split(BGR_Frame)
		#Reduce built in method to get the maximum of a 3 given matrices
		BRG_Max = np.maximum.reduce([B_Frame, G_Frame, R_Frame])
		BRG_Min = np.minimum.reduce([B_Frame, G_Frame, R_Frame])
		#at uniform daylight
		Rule_1 = np.logical_and.reduce([R_Frame > 95, G_Frame > 40, B_Frame > 20 ,
	                                 BRG_Max - BRG_Min > 15,abs(R_Frame - G_Frame) > 15, 
	                                 R_Frame > G_Frame, R_Frame > B_Frame])
		#the skin colour under flashlight or daylight lateral illumination
		Rule_2 = np.logical_and.reduce([R_Frame > 220, G_Frame > 210, B_Frame > 170,
	                         abs(R_Frame - G_Frame) <= 15, R_Frame > B_Frame, G_Frame > B_Frame])
		#Rule_1 U Rule_2
		RGB_Rule = np.logical_or(Rule_1, Rule_2)
		if plot == True:
			rgb_img = cv2.merge([R_Frame,G_Frame,B_Frame])
			fig = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			fig.suptitle('RGB space', fontsize=16)
			ax1 = fig.add_subplot(1, 2, 1)

			ax1.axes.get_xaxis().set_visible(False)
			ax1.axes.get_yaxis().set_visible(False)
			ax1.set_title('Original-Image')
			ax1.imshow(rgb_img)

			img_bw = RGB_Rule.astype(np.uint8)  #b/w
			img_bw*=255
			ax2 = fig.add_subplot(1, 2, 2)
			ax2.axes.get_xaxis().set_visible(False)
			ax2.axes.get_yaxis().set_visible(False)
			ax2.set_title('RGB-Mask')
			
			ax2.imshow(img_bw,cmap='gray', vmin=0, vmax=255,interpolation='nearest')#grayscale
		return RGB_Rule
	def lines(self,axis):
		#YCbCr
		line1 = 1.5862  * axis + 20
		line2 = 0.3448  * axis + 76.2069
		#Cr ≥ -4.5652 × Cb + 234.5652
		line3 = -1.005 * axis + 234.5652
		line4 = -1.15   * axis + 301.75
		line5 = -2.2857 * axis + 432.85
		return [line1,line2,line3,line4,line5]
	def Rule_B(self,YCrCb_Frame,plot=False):
		'''this function implements the five bounding rules of Cr-Cb components
		--inputs: 
		YCrCb_Frame: YCrCb components of an image
		plot: Bool type variable,if set to True draw the output of the algorithm
		--return a anumpy array of type bool
		'''
		Y_Frame,Cr_Frame, Cb_Frame = [YCrCb_Frame[...,YCrCb] for YCrCb in range(3)]
		line1,line2,line3,line4,line5 = self.lines(Cb_Frame)
		YCrCb_Rule = np.logical_and.reduce([line1 - Cr_Frame >= 0,
											line2 - Cr_Frame <= 0,
											line3 - Cr_Frame <= 0,
											line4 - Cr_Frame >= 0,
											line5 - Cr_Frame >= 0])
		if plot == True:
			fig1 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig1.add_subplot(1, 1, 1)
			ax1.scatter(Cb_Frame, Cr_Frame, alpha=0.8, c='black', edgecolors='none', s=10, label="Cr")
			ax1.set_xlim([0, 255])
			ax1.set_ylim([0, 255])
			ax1.set_xlabel('Cb')
			ax1.set_ylabel('Cr')
			ax1.xaxis.set_label_coords(0.5, -0.025)

			x_axis = np.linspace(0, 255,100)
			line1,line2,line3,line4,line5 = self.lines(x_axis)
			ax1.plot(x_axis, line1, alpha=0.5, c='b', label="line1")
			ax1.plot(x_axis, line2, alpha=0.5, c='g', label="line2")
			ax1.plot(x_axis, line3, alpha=0.5, c='r', label="line3")
			ax1.plot(x_axis, line4, alpha=0.5, c='m', label="line4")
			ax1.plot(x_axis, line5, alpha=0.5, c='y', label="line5")
			plt.title('Bounding Rule for Cb-Cr space')
			plt.legend(loc=(1,0.7))
			#plot the Y Cr and Cb components on a different figure
			fig2 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			fig2.suptitle('Y-Cr-Cb components', fontsize=16)
			#Y components
			ax2 = fig2.add_subplot(3, 1, 1)
			ax2.set_title('Distribution of Y')
			ax2.title.set_position([0.9, 0.95])
			ax2.set_xlabel('pixel intensity')
			ax2.xaxis.set_label_coords(0.5, -0.025)
			ax2.set_ylabel('number of pixels')
			ax2.hist(Y_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')	
			#Cb components
			ax3 = fig2.add_subplot(3, 1, 2)
			ax3.set_title('Distribution of Cb')
			ax3.title.set_position([0.9, 0.95])
			ax3.set_xlabel('pixel intensity')
			ax3.xaxis.set_label_coords(0.5, -0.025)
			ax3.set_ylabel('number of pixels')
			ax3.hist(Cb_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')
			#Cr components
			ax4 = fig2.add_subplot(3, 1, 3)
			ax4.set_title('Distribution of Cr')
			ax4.title.set_position([0.9, 0.95])
			ax4.set_xlabel('pixel intensity')
			ax4.xaxis.set_label_coords(0.5, -0.025)
			ax4.set_ylabel('number of pixels')
			ax4.hist(Cr_Frame.ravel(), bins=256, range=(0, 256), fc='b', ec='b')
			#show the effect of the bounding rules of Cr-Cb 
			img_bw = YCrCb_Rule.astype(np.uint8)  #b/w
			img_bw*=255
			fig3 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig3.add_subplot(1, 1, 1)
			ax1.axes.get_xaxis().set_visible(False)
			ax1.axes.get_yaxis().set_visible(False)
			ax1.set_title('CrCb-Mask')
			
			ax1.imshow(img_bw,cmap='gray', vmin=0, vmax=255,interpolation='nearest')#grayscale
		return YCrCb_Rule
	def Rule_C(self,HSV_Frame,plot=False):
		'''this function implements the five bounding rules of Cr-Cb components
		--inputs: 
		HSV_Frame: Hue,saturation and value components of a given image
		plot: Bool type variable,if set to True draw the output of the algorithm
		--return a anumpy array of type bool
		'''
		Hue,Sat,Val = [HSV_Frame[...,i] for i in range(3)]
		HSV_ = np.logical_or(Hue < 50, Hue > 150)
		if plot == True:
			#Plot Hue(x_axis) vs Value(y_axis)
			fig1 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig1.add_subplot(1, 2, 1)
			ax1.scatter(Hue, Val, alpha=0.8, c='b', edgecolors='none', s=10, label="Cr")
			ax1.set_xlim([0, 255])
			ax1.set_ylim([0, 255])
			ax1.set_xlabel('Hue')
			ax1.set_ylabel('Val')
			ax1.set_title('HSV skin color Distribution H vs V')
			#Plot Hue(x_axis) vs Sat(y_axis)
			ax2 = fig1.add_subplot(1, 2, 2)
			ax2.set_title('HSV skin color Distribution H vs S')
			ax2.scatter(Hue, Sat, alpha=0.8, c='b', edgecolors='none', s=10, label="Cr")
			ax2.set_xlim([0, 255])
			ax2.set_ylim([0, 255])
			ax2.set_xlabel('Hue')
			ax2.set_ylabel('Sat')
			#plot Hue mask
			Hue_bw = HSV_.astype(np.uint8)  #Test the output
			Hue_bw*=255
			fig2 = plt.figure(figsize=(9, 8), dpi=90, facecolor='w', edgecolor='k')
			ax1 = fig2.add_subplot(1, 1, 1)
			ax1.axes.get_xaxis().set_visible(False)
			ax1.axes.get_yaxis().set_visible(False)
			ax1.set_title('Hue-Mask')
			#plot as a Grayscale image
			ax1.imshow(Hue_bw,cmap='gray', vmin=0, vmax=255,interpolation='nearest')
		return HSV_

	def RGB_H_CbCr(self, Frame_,plot=False):
		'''this function implements the RGB_H_CbCr bounding rule
		--inputs: 
		Frame_: BGR image
		plot: Bool type variable,if set to True draw the output of the algorithm
		--return a anumpy array of type integer
		'''
		Ycbcr_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2YCrCb)
		HSV_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2HSV)
		#Rule A ∩ Rule B ∩ Rule C
		skin_ = np.logical_and.reduce([self.Rule_A(Frame_), self.Rule_B(Ycbcr_Frame), self.Rule_C(HSV_Frame)])
		if plot == True:
			skin_bw= skin_.astype(np.uint8) 
			skin_bw*=255

			RGB_Frame = cv2.cvtColor(Frame_, cv2.COLOR_BGR2RGB)#original
			seg = cv2.bitwise_and(Frame_,Frame_,mask=skin_bw)

			cv2.imshow("Extracted Skin",seg)#graycsale image
		return np.asarray(skin_, dtype=np.uint8)

if __name__ == "__main__":
	# if len(sys.argv) < 2:
	# 	print("please give me an image !!!")
	# 	sys.exit(0)
	# image = sys.argv[1]
	image = './images/test_img.jpg'
	try:
		img = np.array(cv2.imread(image), dtype=np.uint8)
	except:
		print('Error while loading the Image,image does not exist!!!!')
		sys.exit(1)
	test = Skin_Detect()
	YCrCb_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	HSV_Frames = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	test.Rule_A(img,True)
	test.Rule_B(YCrCb_Frames,True)
	test.Rule_C(HSV_Frames,True)
	test.RGB_H_CbCr(img,True)
	plt.show()
	cv2.waitKey(0)
