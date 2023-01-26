from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from hdbscan import HDBSCAN

class Clusterizer:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def fit_transform(self, data):
        if self.method == 'kmeans':
            clusterer = KMeans(**self.kwargs)
            return clusterer.fit_predict(data)
        elif self.method == 'agglomerative':
            clusterer = AgglomerativeClustering(**self.kwargs)
            return clusterer.fit_predict(data)
        elif self.method == 'dbscan':
            clusterer = DBSCAN(**self.kwargs)
            return clusterer.fit_predict(data)
        else:
             raise ValueError("Invalid method. Choose 'kmeans', 'agglomerative', or 'dbscan'.")
        # elif self.method == 'hdbscan':
        #     clusterer = HDBSCAN(**self.kwargs)
        #     return clusterer.fit_predict(data)
        # else:
        #     raise ValueError("Invalid method. Choose 'kmeans', 'agglomerative', 'dbscan' or 'hdbscan'.")