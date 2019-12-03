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
            break

def eval_ICA(vectors,using_shape, featureCounts):
    ones = np.ones((vectors.shape[1],1),dtype = vectors.dtype)
    ones = ones/vectors.shape[1]

    fossil_mean = np.matmul(vectors, ones)
    zero_mean_fossil = vectors - fossil_mean

    for fc in featureCounts:
        w, z = PCA(zero_mean_fossil, num_features=fc)
        y, w_ica, conv = ICA(z, learning_rate=1e-3)

        w_features = np.linalg.pinv(w_ica.dot(w))

        # Recreate the original data using the ica features and the ICA shifted data y
        reconstructed_vector= w_features.dot(y) + zero_mean_fossil

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
            break


def eval_NMF(vectors,using_shape, featureCounts):

    for fc in featureCounts:
        W_nmf, H = NMF(vectors,num_features = fc)

        reconstructed_vector = W_nmf.dot(H)

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
            break


