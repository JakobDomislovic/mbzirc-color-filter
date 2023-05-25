import tkinter as tk
from tkSliderWidget import Slider

def print_hue(hue):
    print(hue[0])
    print(hue[1])

root = tk.Tk()

slider = Slider(root, width = 400, height = 60, min_val = 0, max_val = 180, init_lis = [0,180], show_value = True)
slider.pack()

slider2 = Slider(root, width = 400, height = 60, min_val = 0, max_val = 255, init_lis = [0, 255], show_value = True)
slider2.pack()

slider3 = Slider(root, width = 400, height = 60, min_val = 0, max_val = 255, init_lis = [0, 100], show_value = True)
slider3.pack()


# optionally add a callback on value change
slider.setValueChageCallback(print_hue)
slider2.setValueChageCallback(lambda vals: print("Saturation: ", vals))
slider3.setValueChageCallback(lambda vals: print("Value: ", vals))

root.title("HUE/SATURATION/VALUE SLIDE")
root.mainloop()

#print(slider.getValues())
