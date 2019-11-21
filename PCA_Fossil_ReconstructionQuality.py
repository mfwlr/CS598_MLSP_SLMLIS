import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import  Image
from pathlib import Path

def PCA(data, num_features=3):
    # Get the covariance and the eigenvalues/vectors of that matrix
    Cov = np.dot(data, data.T) / (data.shape[1])

    eigv, eigvec = linalg.eigsh(Cov, k=num_features,which='LM')

    # Select my top num_features eigenvectors/values

    selected_eigvec = eigvec[:, :num_features]
    selected_eigvals = eigv[:num_features]

    # Calculate the diagonal/sqrt eigenvalues, w, and z
    diagEvals = np.diag(np.sqrt(selected_eigvals))

    w = np.matmul(np.linalg.inv(diagEvals), selected_eigvec.T)

    z = np.dot(w, data)

    return w, z

def read_images(count=10):

    ds_directory =  Path.cwd().joinpath('Fossil Data Sets')
    dirs = ds_directory.glob('*')
    vectorized =  None
    c = 0
    for path in dirs:
        files = path.glob("*")
        for im in files:
            if c == count:
                return vectorized, using_shape

            img = Image.open(im)
            #Get rid of this eventually in favor of cutting imgs into chunks
            img.thumbnail((75, 75))
            #Scale as float
            image = np.array(img, np.float)/255

            if type(vectorized) ==  type(None):
                x,y,z = image.shape
                vectorized = np.zeros((x*y*z,1),dtype=image.dtype)
                using_shape = (x,y,z)

            try:
                vectorized = np.concatenate(
                (vectorized, np.reshape(image.flatten('F')[:vectorized.shape[0]],(-1,1))), axis=1)
                c+=1
            except:
                ''''''




def eval_PCA(vectors,using_shape, featureCounts):
    ones = np.ones((vectors.shape[1],1),dtype = vectors.dtype)
    ones = ones/vectors.shape[1]

    fossil_mean = np.matmul(vectors, ones)
    zero_mean_fossil = vectors - fossil_mean
    print(vectors.shape)
    for fc in featureCounts:
        w, z = PCA(zero_mean_fossil, num_features=fc)

        reconstructed_vector = np.linalg.pinv(w).dot(z)+zero_mean_fossil

        for col in range(vectors.shape[1]):
            fossil =  reconstructed_vector[:,col]
            real_fossil = vectors[:,col]
            print(fossil)
            fossil = np.reshape(fossil, using_shape, 'F')
            real_fossil = np.reshape(real_fossil, using_shape, 'F')
            feature_figure, axes = plt.subplots(1, 2, sharey=True)
            axes[0].imshow(real_fossil)
            axes[1].imshow(fossil[:,:,1])
            plt.show()


v, us =  read_images(50)
eval_PCA(v,us,[5])

