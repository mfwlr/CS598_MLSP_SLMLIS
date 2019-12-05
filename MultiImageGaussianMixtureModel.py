from PIL import Image
import numpy as np
from pathlib import Path
import scipy.stats
import matplotlib.pyplot as plt

#Gaussian Mixture Model with tied covariance matrices
rows = []
cols = []
filenames = []
data_list = []
limit = 100
for filename in Path('Fossil Data Sets/MTN-1 squares-JPG').rglob('*.jpg'):
    img = Image.open(filename)
    pix = np.array(img)
    filenames.append(str(filename))
    rows.append(pix.shape[0])
    cols.append(pix.shape[1])
    data_list.append(pix.reshape((-1, 3)) / 255)
    limit -= 1
    if limit <= 0:
        break

data = np.vstack(np.array(data_list))
print("Finished loading data")
mean_data = np.mean(data, axis=1)

for num_clusters in range(5, 6):
    print("Starting cluster: ", num_clusters)
    #initialize parameters with random uniform [0, 1]
    means = np.random.uniform(0, 1, (num_clusters, 3))
    prior = np.random.uniform(0, 1, num_clusters)
    prior /= np.sum(prior)
    cov_list = []
    for cluster in range(num_clusters):
        cov_list.append(np.identity(3))
    covariance = np.sum(np.array(cov_list), axis=0)

    posterior_probs = []
    old_posterior_probs = []
    for iteration in range(20):
        print("Starting iteration: ", iteration)
        # E-step, estimate probability of each pixel being in each class given
        # the current mean/covariance for that class
        all_normals = 0
        tmp_probs = []
        for cluster in range(num_clusters):
            tmp_probs.append(prior[cluster] * scipy.stats.multivariate_normal.pdf(data, means[cluster], covariance))
            all_normals += tmp_probs[-1]

        if posterior_probs is not None:
            old_posterior_probs = posterior_probs
        posterior_probs = []
        for cluster in range(num_clusters):
            posterior_probs.append(np.nan_to_num(tmp_probs[cluster] / all_normals))


        if old_posterior_probs:
            delta = np.sum(np.abs(np.array(old_posterior_probs) - np.array(posterior_probs)))
            print("Delta = ", delta)
            if delta < 0.01:
                break

        # M-step: Recalculate the mean/covariance for each class given the
        # current pixel probabilities for that class
        posterior_sum = []
        posterior_total = 0
        for cluster in range(num_clusters):
            posterior_sum.append(np.sum(posterior_probs[cluster]))
            posterior_total += posterior_sum[-1]

        for cluster in range(num_clusters):
            prior[cluster] = posterior_sum[cluster] / posterior_total

        for cluster in range(num_clusters):
            tmp1 = np.multiply(posterior_probs[cluster][:, None], data)
            tmp2 = np.sum(tmp1, axis=0)
            means[cluster] = tmp2 / posterior_sum[cluster]

        for cluster in range(num_clusters):
            tmp = np.multiply(posterior_probs[cluster][:,None], data - means[cluster])
            cov_list[cluster] = tmp.T @ tmp / tmp.shape[0]
        #make it a tied covariance:
        covariance = np.sum(np.array(cov_list), axis=0)

    start = 0
    for image in range(len(filenames)):
        end = start + rows[image] * cols[image]
        for cluster in range(num_clusters):
            plt.imshow(posterior_probs[cluster][start : end].reshape(rows[image], cols[image]))
            #plt.show()
            plt.savefig('Output/' + filenames[image].replace('.jpg', f"_{cluster}.jpg"), dpi=200)
        start += rows[image] * cols[image]
