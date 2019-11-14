
import numpy as np
from PIL import  Image
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as  plt

from sklearn import cluster

def process_data_sets(k, x_red=1.0/12, y_red=1.0/12):

    ds_directory =  Path.cwd().joinpath('Fossil Data Sets')
    dirs = ds_directory.glob('*')
    for path in dirs:
        files = path.glob("*")
        for im in files:
            img = Image.open(im)
            #Reduce image size to speed computation
            width, height = img.size
            ow =  width
            oh = height
            width *= x_red
            height *=  y_red
            img.thumbnail((width,height))
            image = np.array(img, np.int32)
            x,y,z = image.shape
            reshaped  =  image.reshape(x*y,z)
            kmeans_cluster = cluster.KMeans(n_clusters = k)

            kmeans_cluster.fit(reshaped)
            cluster_centers  = kmeans_cluster.cluster_centers_
            cluster_labels =  kmeans_cluster.labels_
            print("Done with ", im)

            plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
            plt.show()

process_data_sets(6)