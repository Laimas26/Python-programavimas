import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

 
 
def loadData():
     digits = load_digits()
     data = scale(digits.data)
     
     return digits, data
 
def main():
    # global digits, data
    digits, data = loadData()
     
    # global y
    # global clf
    y = digits.target
    
    # global k
    k = len(np.unique(y))
    
    samples, features = data.shape
    
# k-means++ isdelioja centroidus labiau logiskai, taip pagreitinant apsimokyma
# Galimai isdelioja vienodais atstmais vienas nuo kito 
# clf = KMeans(n_clusters=k, init='k-means++', n_init=)
    clf = KMeans(n_clusters=k, init='random', n_init=10)
    
    bench_k_means(clf, '1', data)
    # visualizeData()


def visualizeData():
    plt.figure(figsize=(8, 6))
    colors = plt.cm.nipy_spectral(clf.labels_.astype(float) / k)
    plt.scatter(data[:, 0], data[:, 1], c=colors, marker='.')
    plt.title("K-Means Clustering")
    plt.show()

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    
 
 
main()