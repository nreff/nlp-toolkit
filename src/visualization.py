import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#import umap.umap_ as umap

class Visualizer:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def fit_transform(self, data, labels):
        if self.method == 'scatter':
            plt.scatter(data[:,0], data[:,1], c=labels)
            plt.show()
        elif self.method == 'tsne':
            tsne = TSNE(**self.kwargs)
            reduced_data = tsne.fit_transform(data)
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
            plt.show()
        else:
            raise ValueError("Invalid method. Choose 'scatter', or 'tsne'.")
        # elif self.method == 'umap':
        #     reducer = umap.UMAP(**self.kwargs)
        #     reduced_data = reducer.fit_transform(data)
        #     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
        #     plt.show()
        # else:
        #     raise ValueError("Invalid method. Choose 'scatter', 'tsne' or 'umap'.")