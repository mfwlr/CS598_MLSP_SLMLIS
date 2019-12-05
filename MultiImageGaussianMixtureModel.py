from PIL import Image
import numpy as np
from pathlib import Path
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Gaussian Mixture Model with tied covariance matrices
rows = []
cols = []
filenames = []
data_list = []
limit = 100
num_iterations = 10

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
num_clusters = 5

print("Starting clusters: ", num_clusters)
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
for iteration in range(num_iterations):
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

#Separate image file saving
# start = 0
# for image in range(len(filenames)):
#     end = start + rows[image] * cols[image]
#     for cluster in range(num_clusters):
#         plt.imshow(posterior_probs[cluster][start : end].reshape(rows[image], cols[image]))
#         #plt.show()
#         plt.savefig('Output/' + filenames[image].replace('.jpg', f"_{cluster}.jpg"), dpi=200)
#     start += rows[image] * cols[image]

#Combine probs into a single image with different colors (assumes 5 clusters so has 5 colors)
colors = np.array([[0, 135, 68], [214, 45, 32], [255, 167, 0], [153, 51, 255], [0, 87, 231]]) / 255
start = 0
for image in range(len(filenames)):
    end = start + rows[image] * cols[image]
    image_data = np.empty((num_clusters, rows[image] * cols[image], 3))
    for cluster in range(num_clusters):
        tmp_r = np.multiply(colors[cluster][0], posterior_probs[cluster][start : end])
        tmp_g = np.multiply(colors[cluster][1], posterior_probs[cluster][start : end])
        tmp_b = np.multiply(colors[cluster][2], posterior_probs[cluster][start : end])
        image_data[cluster] = np.dstack((tmp_r, tmp_g, tmp_b))
    combined_data = np.mean(image_data, axis=2)
    max_cluster = np.argmax(combined_data, axis=0)
    pixels = np.empty((rows[image],  cols[image], 3))
    for x in range(rows[image]):
        for y in range(cols[image]):
            for color in range(3):
                pixels[x, y] = image_data[max_cluster[x*cols[image] + y], x*cols[image] + y]
    tmp_max = np.max(pixels.flatten())
    plt.imshow(pixels / tmp_max)
    #plt.show()
    red = mpatches.Patch(color=tuple(colors[0]), label='Class 1')
    green = mpatches.Patch(color=tuple(colors[1]), label='Class 2')
    blue = mpatches.Patch(color=tuple(colors[2]), label='Class 3')
    purple = mpatches.Patch(color=tuple(colors[3]), label='Class 4')
    yellow = mpatches.Patch(color=tuple(colors[4]), label='Class 5')
    plt.legend(handles=[red, green, blue, purple, yellow])
    plt.savefig('Output/' + filenames[image].replace('.jpg', f"_combined.jpg"), dpi=600)
    start += rows[image] * cols[image]

