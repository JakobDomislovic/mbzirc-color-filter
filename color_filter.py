import os
import argparse
import cv2
import time
import glob
import _thread
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from tkSliderWidget import Slider
import matplotlib.pyplot as plt
  
class ColorFilter:

    def __init__(self, video=False) -> None:
        # images path
        self.images_path = './data/from_phone_original/'
        self.image_names = sorted(os.listdir(self.images_path))
        self.images = [os.path.join(self.images_path, img_name) for img_name in self.image_names]
        # Gausian blurr kernel size ---> TODO: in config
        self.blurr_kernel = (25, 25)
        # dilation and erosion kernel size ---> TODO: in config
        self.dilation_erosion_kernel = (25, 25)
        # initial hue boundaries
        self.hue_lower = 0
        self.hue_upper = 180
        # initial saturation boundaries
        self.saturation_lower = 0
        self.saturation_upper = 255
        # initial value boundaries
        self.value_lower = 0
        self.value_upper = 100
        # images
        self.img = cv2.imread(self.images[0])
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        h, w, c = self.img.shape
        self.mask = np.zeros([h,w,c],dtype=np.uint8)
        self.result = np.zeros([h,w,c],dtype=np.uint8)
        
        # tkinter gui
        self.root = tk.Tk()
        self.init_gui()
        self.slider_widget()
        self.image_widget()
        self.image_hsv_widget()
        self.mask_widget()
        self.result_widget()
        
        self.video = video
        if self.video:
            self.cap = cv2.VideoCapture('/dev/video2')
            # Check if the webcam is opened correctly
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam")

        else:
            self.image_list_widget()

    def run(self):
        if not self.video:
            self.root.mainloop()
        else:
            while True:
                self.root.update_idletasks()
                self.root.update()
                ret, frame = self.cap.read()
                self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

                resized = cv2.resize(self.img, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
                self.img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
                resized = cv2.resize(self.img_hsv, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
                self.img_hsv_tk = ImageTk.PhotoImage(Image.fromarray(resized))

                self.label_img.configure(image=self.img_tk)
                self.label_img.image = self.img_tk

                self.label_img_hsv.configure(image=self.img_hsv_tk)
                self.label_img_hsv.image = self.img_hsv_tk

                self.filter()

    def filter(self):
        # ---- PREPROCESSING -----
        # set boundaries based on sliders
        lower_boundaries,upper_boundaries = self.set_boundaries()
        # blurring image --> filter shape depends on image shape
        blurred = cv2.GaussianBlur(self.img_hsv, self.blurr_kernel, 0)
        
        # ---- FINDING MASK AND CONTOURS -----
        # extract only black color
        mask = cv2.inRange(blurred, lower_boundaries, upper_boundaries)
        # dilation and erosion of mask
        mask  = self.dilation_and_erosion(mask)
        result = cv2.bitwise_and(self.img, self.img, mask = mask)
        # change result to BGR so we can see it better
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)        
        
        # find contours
        contours, hierarchy, hull_list = self.find_contours(mask, result)
        # find two biggest contours    
        if len(contours) > 0:
            # find index of largest contour
            largest_contour_index = self.largest_contour(hull_list)
            # draw contours
            cv2.drawContours(result, hull_list, largest_contour_index, (0, 255, 0), 5)
            # fit rectangle
            x,y,w,h = self.fit_rectangle_to_contour(hull_list[largest_contour_index])
            # draw rectangle
            cv2.rectangle(result, (x,y), (x+w,y+h), (255,0,0), 5)
            # find COA
            result = cv2.circle(result, (x+w//2,y+h//2), radius=20, color=(250, 250, 0), thickness=-1)

        print(lower_boundaries)
        print(upper_boundaries)

        # ---- PLOT ----
        # save mask and result to class instance variables
        resized_mask = cv2.resize(mask, (250,250), interpolation = cv2.INTER_AREA)
        resized_result = cv2.resize(result, (250,250), interpolation = cv2.INTER_AREA)
        
        self.mask_tk = ImageTk.PhotoImage(Image.fromarray(resized_mask))
        self.result_tk = ImageTk.PhotoImage(Image.fromarray(resized_result))

        self.label_mask.configure(image=self.mask_tk)
        self.label_mask.image = self.mask_tk
        
        self.label_result.configure(image=self.result_tk)
        self.label_result.image = self.result_tk
    
    def set_boundaries(self):
        # find mask based on boundaries value
        lower_boundaries = np.array([self.hue_lower, self.saturation_lower, self.value_lower])
        upper_boundaries = np.array([self.hue_upper, self.saturation_upper, self.value_upper])
        
        return lower_boundaries, upper_boundaries

    def dilation_and_erosion(self, mask):
        mask_dilate = cv2.dilate(mask, self.dilation_erosion_kernel, iterations=1)
        mask_erosion = cv2.erode(mask_dilate, self.dilation_erosion_kernel, iterations=1)
        return mask_erosion
        
    def find_contours(self, mask, result):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # RETR_CCOMP finds all contours
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        return contours, hierarchy, hull_list

    def largest_contour(self, hull_list):
        # calculate area
        cont_area = []
        for i in range(len(hull_list)):
            area = cv2.contourArea(hull_list[i])
            cont_area.append(area)
        largest_contour_index = cont_area.index(max(cont_area))
        return largest_contour_index

    def fit_rectangle_to_contour(self, contour):
        (x,y,w,h) = cv2.boundingRect(contour)
        return x,y,w,h

    def init_gui(self):
        self.root.geometry("800x800")
        self.root.title("COLOR FILTER")

    def slider_widget(self):
        frame = tk.Frame(self.root, width=400, height=200)
        frame.pack()
        frame.place(anchor='center', x=400, y=100)
        # hue
        slider = Slider(frame, width = 400, height = 60, min_val = 0, max_val = 179, init_lis = [0,179], show_value = True, addable=True, removable=True)
        slider.pack()
        # saturation
        slider2 = Slider(frame, width = 400, height = 60, min_val = 0, max_val = 255, init_lis = [0, 255], show_value = True, addable=True, removable=True)
        slider2.pack()
        # value
        slider3 = Slider(frame, width = 400, height = 60, min_val = 0, max_val = 255, init_lis = [0, 100], show_value = True, addable=True, removable=True)
        slider3.pack()

        # optionally add a callback on value change
        slider.setValueChageCallback(self.set_hue_callback)
        slider2.setValueChageCallback(self.set_saturation_callback)
        slider3.setValueChageCallback(self.set_value_callback)

    def image_widget(self):
        frame = tk.Frame(self.root, width=250, height=250)
        frame.pack()
        frame.place(anchor='center', x=250, y=400)
        # resize for tkinter
        resized = cv2.resize(self.img, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        
        ## Create a Label Widget to display the text or Image
        self.label_img = tk.Label(frame, image = self.img_tk)
        self.label_img.pack()

    def image_hsv_widget(self):
        frame = tk.Frame(self.root, width=250, height=250)
        frame.pack()
        frame.place(anchor='center', x=500, y=400)
        # resize for tkinter
        resized = cv2.resize(self.img_hsv, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.img_hsv_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        ## Create a Label Widget to display the text or Image
        self.label_img_hsv = tk.Label(frame, image = self.img_hsv_tk)
        self.label_img_hsv.pack()
    
    def mask_widget(self):
        frame = tk.Frame(self.root, width=250, height=250)
        frame.pack()
        frame.place(anchor='center', x=250, y=650)
        # resize for tkinter
        resized = cv2.resize(self.mask, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.mask_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        ## Create a Label Widget to display the text or Image
        self.label_mask = tk.Label(frame, image = self.mask_tk)
        self.label_mask.pack()

    def result_widget(self):
        frame = tk.Frame(self.root, width=250, height=250)
        frame.pack()
        frame.place(anchor='center', x=500, y=650)
        # resize for tkinter
        resized = cv2.resize(self.result, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.result_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        ## Create a Label Widget to display the text or Image
        self.label_result = tk.Label(frame, image = self.result_tk)
        self.label_result.pack()
    
    def load_new_image(self, x):
        img = self.images[self.lst.curselection()[0]]
        self.img = cv2.imread(img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        
        resized = cv2.resize(self.img, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        resized = cv2.resize(self.img_hsv, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.img_hsv_tk = ImageTk.PhotoImage(Image.fromarray(resized))

        self.label_img.configure(image=self.img_tk)
        self.label_img.image = self.img_tk
        
        self.label_img_hsv.configure(image=self.img_hsv_tk)
        self.label_img_hsv.image = self.img_hsv_tk
        # in the end call filter() method
        self.filter()
    
    def image_list_widget(self):
        # images scrollbar and listbox
        frame = tk.Frame(self.root, width=40, height=200)
        frame.pack()
        frame.place(x=0, y=0)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y') 
        
        self.lst = tk.Listbox(frame)
        self.lst.pack()
        for fname in self.image_names:
            self.lst.insert(tk.END, fname)
        self.lst.bind("<<ListboxSelect>>", self.load_new_image)
        
        # list and scrollbar interaction
        self.lst.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.lst.yview)
        
    def set_hue_callback(self, data):
        self.hue_lower = int(data[0])
        self.hue_upper = int(data[1])
        self.filter()

    def set_saturation_callback(self, data):
        self.saturation_lower = int(data[0])
        self.saturation_upper = int(data[1])
        self.filter()
        
    def set_value_callback(self, data):
        self.value_lower = int(data[0])
        self.value_upper = int(data[1])
        self.filter()

    def show_img_hsv_mask_result(self):
        self.ax[0].imshow(self.img)
        self.ax[0].set_title(f'RGB', fontsize=self.fontsize)

        self.ax[1].imshow(self.img_hsv)
        self.ax[1].set_title(f'HSV', fontsize=self.fontsize)

        self.ax[2].imshow(self.mask, cmap='gray')
        self.ax[2].set_title(f'Mask', fontsize=self.fontsize)

        self.ax[3].imshow(self.result)
        self.ax[3].set_title(f'Result', fontsize=self.fontsize)
        plt.show()


def parse_args():
    parser=argparse.ArgumentParser(description="Select if you want to stream video.")
    parser.add_argument("-v", "--video", default=False)
    args=parser.parse_args()
    return args

if __name__ == "__main__":
    inputs=parse_args()    
    filter = ColorFilter(video=inputs.video)
    filter.run()
