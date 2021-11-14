import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as LE

f = open("swissroll-data.txt", 'r')
data = []
for line in f.readlines():
    line = line.strip()
    data.append([float(i) for i in line.split()])
data = np.array(data)

feature = np.array(data)[:, 1:]
label = np.array(data)[:, 0]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(feature)
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=0.7, c=label)
plt.title("PCA")
plt.savefig("swissroll-pca.png", dpi=300)
plt.clf()

lda = LDA(n_components=2)
lda_result = lda.fit_transform(feature, label)
plt.scatter(lda_result[:, 0], lda_result[:, 1], s=0.7, c=label)
plt.title("LDA")
plt.savefig("swissroll-lda.png", dpi=300)
plt.clf()

kpca = KernelPCA(n_components=2, kernel="rbf")
kpca_result = kpca.fit_transform(feature, label)
plt.scatter(kpca_result[:, 0], kpca_result[:, 1], s=0.7, c=label)
plt.title("KPCA")
plt.savefig("swissroll-kpca.png", dpi=300)
plt.clf()

isomap = Isomap(n_components=2)
isomap_result = isomap.fit_transform(feature)
plt.scatter(isomap_result[:, 0], isomap_result[:, 1], s=0.7, c=label)
plt.title("Isomap")
plt.savefig("swissroll-isomap.png", dpi=300)
plt.clf()

lle = LLE(n_components=2, n_neighbors=30)
lle_result = lle.fit_transform(feature, label)
plt.scatter(lle_result[:, 0], lle_result[:, 1], s=0.7, c=label)
plt.title("LLE")
plt.savefig("swissroll-lle.png", dpi=300)
plt.clf()

le = LE(n_components=2, n_neighbors=20)
le_result = le.fit_transform(feature, label)
plt.scatter(le_result[:, 0], le_result[:, 1], s=0.7, c=label)
plt.title("Laplacian Eigenmaps")
plt.savefig("swissroll-le.png", dpi=300)
plt.clf()
