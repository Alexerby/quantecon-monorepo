import matplotlib.pyplot as plt


def compare_reconstructions(model, X_samples, img_shape=(28, 28)):
    """
    Independent of the model class.
    It just needs a 'predict' method and a numpy array.
    """
    reconstructed = model._predict(X_samples)

    n = len(X_samples)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))

    for i in range(n):
        # Original
        axes[0, i].imshow(X_samples[i].reshape(img_shape), cmap="gray")
        axes[0, i].axis("off")

        # Reconstruction
        axes[1, i].imshow(reconstructed[i].reshape(img_shape), cmap="gray")
        axes[1, i].axis("off")

    plt.show()
