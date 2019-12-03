from PIL import Image
import numpy as np
from pathlib import Path
import scipy.stats
import matplotlib.pyplot as plt

#Gaussian Mixture Model with tied covariance matrices 
for filename in Path('Fossil Data Sets').rglob('*.jpg'):
    img = Image.open(filename)
    pix = np.array(img)
    rows = pix.shape[0]
    cols = pix.shape[1]

    data = pix.reshape((-1, 3)) / 255
    #data = np.vstack(data)
    mean_data = np.mean(data, axis=1)

    print("Finished loading data")

    for num_clusters in range(4, 5):
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
            # E-step
            all_normals = 0
            tmp_probs = []
            for cluster in range(num_clusters):
                tmp_probs.append(prior[cluster] * scipy.stats.multivariate_normal.pdf(data, means[cluster], covariance))
                all_normals += tmp_probs[-1]

            if posterior_probs is not None:
                old_posterior_probs = posterior_probs
            posterior_probs = []
            for cluster in range(num_clusters):
                posterior_probs.append(tmp_probs[cluster] / all_normals)

            if old_posterior_probs:
                delta = np.sum(np.abs(np.array(old_posterior_probs) - np.array(posterior_probs)))
                print("Delta = ", delta)
                if delta < 0.01:
                    break

            # M-step
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

        for cluster in range(num_clusters):
            plt.imshow(posterior_probs[cluster].reshape(rows, cols))
            plt.show()
    break