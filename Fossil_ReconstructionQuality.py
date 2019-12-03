import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import  Image
from pathlib import Path
import matplotlib.gridspec as gridspec


def g(u):
    return np.tanh(u)

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


def ICA(data, num_iterations=500, learning_rate=0.00001, threshold=0.001):
    # We want a data shaped matrix out, which means our random w must have:
    w_ica = np.identity(data.shape[0], dtype=data.dtype)

    y = w_ica.dot(data)

    N = y.shape[1]
    I = N * np.diag(np.ones(data.shape[0]))

    curI = 0
    # Throw away "too large" norm
    calc_norm = 1000

    # The Infomax iterative update from the slides with a thresholding consideration
    while calc_norm > threshold and curI < num_iterations:
        innerTerm = np.dot(g(y), y.T)

        subtraction = I - innerTerm
        w_delta = np.dot(subtraction, w_ica)

        new_w = w_ica + learning_rate * w_delta

        calc_norm = np.linalg.norm(new_w - w_ica)

        w_ica = new_w

        y = w_ica.dot(data)

    # Need to find and return the y
    return y, w_ica, False


def NMF(data, num_features=3, num_iterations=100, threshold=0.001):
    # Set up a random W
    W = np.random.rand(data.shape[0], num_features)

    # Estimate our first H
    H = np.random.rand(num_features, data.shape[1])

    H[H < 0] = 0

    # Junk values for the first loop run
    norm_distance_w = 1000
    norm_distance_h = 1000
    nmfiter = 0

    # While we have not yet hit our iteration number and the norms
    # have yet to converge under the threshold, keep on updating W and H
    while norm_distance_w > threshold and norm_distance_h > threshold and nmfiter < num_iterations:
        # Following the multiplicative update from
        # https://mlexplained.com/2017/12/28/a-practical-introduction-to-nmf-nonnegative-matrix-factorization/

        h_up_num = W.T.dot(data)
        h_up_denom = W.T.dot(W.dot(H)) + 1e-6

        new_H = np.multiply(H, (h_up_num / h_up_denom))

        new_H[H < 0] = 0

        w_up_num = data.dot(H.T)
        w_up_denom = W.dot(H.dot(H.T)) + 1e-6

        new_W = np.multiply(W, (w_up_num / w_up_denom))

        new_W[W < 0] = 0

        # Multiplicative update runs until W and H are stable as a threshold, which does
        # not always seem to get the "perfect" norm...
        # https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
        norm_distance_w = np.linalg.norm(new_W - W)
        norm_distance_h = np.linalg.norm(new_H - H)

        W = new_W
        H = new_H
        nmfiter += 1

    return W, H


def read_images(count=10):

    ds_directory =  Path.cwd().joinpath('Fossil Data Sets')
    dirs = ds_directory.glob('*')
    vectorized =  None
    c = 0
    for path in dirs:
        files = path.glob("*")
        for im in files:
            if c == count:
                return vectorized[:,1:], using_shape

            img = Image.open(im)
            #Get rid of this eventually in favor of cutting imgs into chunks
            img.thumbnail((100,100))
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




def eval_chart(vectors,using_shape, featureCounts):
    ones = np.ones((vectors.shape[1], 1), dtype=vectors.dtype)
    ones = ones / vectors.shape[1]

    fossil_mean = np.matmul(vectors, ones)
    zero_mean_fossil = vectors - fossil_mean



    project_figure, axes = plt.subplots(nrows = 4, ncols = len(featureCounts), figsize = (24,16))


    curCol = 0
    for fc in featureCounts:
        W_nmf, H = NMF(vectors,num_features = fc)

        w, z = PCA(zero_mean_fossil, num_features=fc)
        y, w_ica, conv = ICA(z, learning_rate=1e-3)

        w_features = np.linalg.pinv(w_ica.dot(w))

        reconstructed_vector_ica = w_features.dot(y) + zero_mean_fossil

        reconstructed_vector_pca = np.linalg.pinv(w).dot(z) + zero_mean_fossil

        reconstructed_vector_nmf = W_nmf.dot(H)

        fossil_pca = reconstructed_vector_pca[:,0]
        fossil_ica = reconstructed_vector_ica[:, 0]
        fossil_nmf = reconstructed_vector_nmf[:, 0]
        real_fossil = vectors[:, 0]

        fossil_pca = np.reshape(fossil_pca, using_shape, 'F')
        fossil_ica = np.reshape(fossil_ica, using_shape, 'F')
        fossil_nmf = np.reshape(fossil_nmf, using_shape, 'F')
        real_fossil = np.reshape(real_fossil, using_shape, 'F')

        #ax1 = project_figure.add_subplot(p_gs[0,curCol])
        #ax2 = project_figure.add_subplot(p_gs[1,curCol])
        #ax3 = project_figure.add_subplot(p_gs[2,curCol])
        #ax4 = project_figure.add_subplot(p_gs[3,curCol])

        axes[0,curCol].imshow(real_fossil)
        axes[1,curCol].imshow(fossil_pca[:, :, 1])
        axes[2,curCol].imshow(fossil_ica[:, :, 1])
        axes[3,curCol].imshow(fossil_nmf[:, :, 1])

        curCol +=1

    for ax, col in zip(axes[0], featureCounts):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], ['Real Fossil', 'PCA', 'ICA', 'NMF']):
        ax.set_ylabel(row, rotation=0, size="medium")
    project_figure.tight_layout()
    plt.show()


v, us =  read_images(50)
eval_chart(v,us,[2,5])
#eval_PCA(v,us,[2,5,10,15,20,25,30,35,40])
#eval_NMF(v,us,[2,5,10,15,20,25,30,35,40])
#eval_ICA(v,us,[2,5,10,15,20,25,30,35,40])

