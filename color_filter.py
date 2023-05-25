import os
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

    def __init__(self) -> None:
        # images path
        self.images_path = './data/from_phone_original/'
        # Gausian blurr kernel size
        self.blurr = (55, 55)
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
        self.img = cv2.imread(self.images_path + '0020.jpeg')
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
        # self.image_list_widget()

    def run(self):
        self.root.mainloop()

    def filter(self):
        # find mask based on boundaries value
        lower_boundaries = np.array([self.hue_lower, self.saturation_lower, self.value_lower])
        upper_boundaries = np.array([self.hue_upper, self.saturation_upper, self.value_upper])
        # blurring image --> filter shape depends on image shape
        blurred = cv2.GaussianBlur(self.img_hsv, self.blurr, 0)  # TODO: put in config
        # extract only black color
        mask = cv2.inRange(blurred, lower_boundaries, upper_boundaries)
        result = cv2.bitwise_and(self.img, self.img, mask = mask)
        # change result to BGR so we can see it better
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)        
        
        # find contours
        contours, hull_list = self.find_contours(mask, result)

        print(lower_boundaries)
        print(upper_boundaries)
        # save mask and result to class instance variables
        resized_mask = cv2.resize(mask, (250,250), interpolation = cv2.INTER_AREA)
        resized_result = cv2.resize(result, (250,250), interpolation = cv2.INTER_AREA)
        
        self.mask_tk = ImageTk.PhotoImage(Image.fromarray(resized_mask))
        self.result_tk = ImageTk.PhotoImage(Image.fromarray(resized_result))

        self.label_mask.configure(image=self.mask_tk)
        self.label_mask.image = self.mask_tk
        
        self.label_result.configure(image=self.result_tk)
        self.label_result.image = self.result_tk
        
    def find_contours(self, mask, result):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
            cv2.drawContours(result, contours, i, (255, 0, 0), 5)    
            cv2.drawContours(result, hull_list, i, (0, 255, 0), 5)
        return contours, hull_list

    def init_gui(self):
        self.root.geometry("800x800")
        self.root.title("HUE/SATURATION/VALUE SLIDE")

    def slider_widget(self):
        frame = tk.Frame(self.root, width=400, height=200)
        frame.pack()
        frame.place(anchor='center', x=400, y=100)
        # hue
        slider = Slider(frame, width = 400, height = 60, min_val = 0, max_val = 180, init_lis = [0,180], show_value = True, addable=True, removable=True)
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
        label = tk.Label(frame, image = self.img_tk)
        label.pack()

    def image_hsv_widget(self):
        frame = tk.Frame(self.root, width=250, height=250)
        frame.pack()
        frame.place(anchor='center', x=500, y=400)
        # resize for tkinter
        resized = cv2.resize(self.img_hsv, (250,250), interpolation = cv2.INTER_AREA)        # tkinter images
        self.img_hsv_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        ## Create a Label Widget to display the text or Image
        label = tk.Label(frame, image = self.img_hsv_tk)
        label.pack()
    
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
    
    def showimg(self):
        pass
    
    def image_list_widget(self):
        # TODO
        lst = tk.Listbox(self.root)
        lst.pack()
        lst.place(x=0, y=0)
        img_names = sorted(os.listdir(self.images_path))
        namelist = [os.path.join(self.images_path, img_name) for img_name in img_names]
        for fname in namelist:
            lst.insert(tk.END, fname)
        lst.bind("<<ListboxSelect>>", self.showimg)
        
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

        
if __name__ == "__main__":
    filter = ColorFilter()
    # start slider widget in new thread
    #_thread.start_new_thread(filter.slider_widget, ())
    filter.run()
