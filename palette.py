import cv2 as cv
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd

an_image = cv.imread("demo.jpg")

print(an_image.shape)

image_array = np.array(an_image)
image_array = image_array.reshape(an_image.shape[0]*an_image.shape[1],-1)

# print("data image : ",image_array)

image_df = pd.DataFrame(image_array, columns=("red", "green", "blue"))

# print(image_df)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(image_df['red'], image_df['green'], image_df['blue'], c=(image_array))
# plt.show()

kmeans = cluster.KMeans(n_clusters=10).fit(image_df)

palette_colors = kmeans.cluster_centers_

for i,(r,g,b) in enumerate(palette_colors):
    palette_colors[i] = (round(r),round(g),round(b))


print(palette_colors)

an_image = cv.resize(an_image,(500,500))

x = 0
y = 0

for (r,g,b) in palette_colors:
    cv.rectangle(an_image,(0,y),(50,y+50),color=(r,g,b),thickness=-1)
    x += 50
    y += 50

cv.imshow("Image",an_image)
cv.waitKey(0)
