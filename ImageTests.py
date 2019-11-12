from PIL import Image
import numpy as np
from pathlib import Path
import scipy.stats

data = []
for filename in Path('Fossil Data Sets').rglob('*.jpg'):
    #2520 rows, 2513 columns, 3 colors
    #flattened to 2520x2513 data points (rows) with 3 values normalized to [0,1]
    pix = np.array(Image.open(filename)).reshape((-1, 3)) / 255
    data.append(pix)

#Stack all of the images into a single array
data = np.vstack(data)

print("Finished loading data")

for num_clusters in range(2, 5):
    print("Starting cluster: ", num_clusters)
    #initialize parameters with random uniform [0, 1]
    means = np.random.uniform(0, 1, (num_clusters, 3))
    prior = np.random.uniform(0, 1, num_clusters)
    covariance = []
    for cluster in range(num_clusters):
        covariance.append(np.identity(3))

    for iteration in range(4):
        print("Starting iteration: ", iteration)
        # E-step
        all_normals = 0
        tmp_probs = []
        for cluster in range(num_clusters):
            tmp_probs.append(prior[cluster] * scipy.stats.multivariate_normal.pdf(data, means[cluster], covariance[cluster]))
            all_normals += tmp_probs[-1]

        posterior_probs = []
        for cluster in range(num_clusters):
            posterior_probs.append(tmp_probs[cluster] / all_normals)

        # M-step
        posterior_sum = []
        posterior_total = 0
        for cluster in range(num_clusters):
            posterior_sum.append(np.sum(posterior_probs[cluster]))
            posterior_total += posterior_sum[-1]

        for cluster in range(num_clusters):
            prior[cluster] = posterior_sum[cluster] / posterior_total

        for cluster in range(num_clusters):
            means[cluster] = np.sum(np.dot(posterior_probs[cluster], data)) / posterior_sum[cluster]

        for cluster in range(num_clusters):
            covariance[cluster] = np.cov((data - means[cluster]))  #Not right, needs to be a sum of 3x3 cov matrices
