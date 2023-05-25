from ipywidgets import interact, widgets
from IPython.display import display
import matplotlib.pyplot as plt
import warnings
import cv2

from skimage.morphology import disk
from skimage.filters import rank
from skimage.color import rgb2gray


def f(Median_Size):
    selem = disk(int(Median_Size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_median = rank.median(img_gray, selem=selem) 

    ax_neu.imshow(img_median, cmap="gray")
    fig.canvas.draw()
    display(fig)

image = cv2.imread( './data/from_phone_original/' + '0001.jpeg' ) 
img_gray = rgb2gray(image)

fig = plt.figure(figsize=(6, 4))
ax_orig = fig.add_subplot(121) 
ax_neu = fig.add_subplot(122) 

ax_orig.imshow(img_gray, cmap="gray")
ax_neu.imshow(img_gray, cmap="gray")
interact(f, Median_Size=widgets.IntSlider(min=1,max=21,step=2,value=1))

plt.show()
