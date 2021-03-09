from matplotlib import pyplot as plt


def visualize_torch(images, gray=False, n_cols=5, n_rows=1):
    """Visualize samples."""

    fig = plt.figure(figsize = (3*n_cols,3*n_rows))
    for i in range(n_cols*n_rows):
        plt.subplot(n_rows,n_cols,i+1)
        plt.axis('off')
        img = images[i].permute(1, 2, 0).squeeze()
        if gray:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
    return fig