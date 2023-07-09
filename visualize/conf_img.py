#%%

import matplotlib.pyplot as plt
import cv2

#%%

img = cv2.imread('/home/kazuya/hdd/Mitosis_detection/outputs/example_data/Fluo-N2DL-HeLa/180_img2.png')
plt.imshow(img > 125), plt.show()

# %%
