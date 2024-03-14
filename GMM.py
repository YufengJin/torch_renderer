import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, title=None, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax)

    if title:
        ax.set_title(title)

# create datas, 4 clusters
X, y = make_blobs(n_samples=500,
                  centers=4, 
                  cluster_std=2,
                  random_state=2021)


# convert into DataFrame, x,y,cluster
data = pd.DataFrame(X)
data.columns=["X1","X2"]
data["cluster"]=y

if True:
# show synthetic data
    plt.figure(figsize=(9,7))
    sns.scatterplot(data=data, 
                    x="X1",
                    y="X2", 
                    hue="cluster",
                    palette=["red","blue","green", "purple"])
    plt.title("Data Clusters")
    plt.show()

# create GMM model
gmm1 = GaussianMixture(2, covariance_type='full',init_params='k-means++', random_state=0)
gmm2 = GaussianMixture(4, covariance_type='full',init_params='k-means++', random_state=0)
gmm3 = GaussianMixture(6, covariance_type='full',init_params='k-means++', random_state=0)
gmm4 = GaussianMixture(8, covariance_type='full',init_params='k-means++', random_state=0)

# create bayesian GMM model
#gmm1 = BayesianGaussianMixture(2, covariance_type='full', random_state=0)
#gmm2 = BayesianGaussianMixture(4, covariance_type='full', random_state=0)
#gmm3 = BayesianGaussianMixture(6, covariance_type='full', random_state=0)
#gmm4 = BayesianGaussianMixture(8, covariance_type='full', random_state=0)

gmm1.fit(X)
gmm2.fit(X)
gmm3.fit(X)
gmm4.fit(X)


# predict label 
#labels = gmm.predict(X)
#data["gmm_predicted_cluster"]=labels

fig, ax=plt.subplots(2,2)
plot_gmm(gmm1, X, ax=ax[0][0], title="2 component GMM")
plot_gmm(gmm2, X, ax=ax[0][1], title="4 component GMM")
plot_gmm(gmm3, X, ax=ax[1][0], title="6 component GMM")
plot_gmm(gmm4, X, ax=ax[1][1], title="8 component GMM")
plt.suptitle("Bayesian GMM with different n components trained on the same data (4 clusters)")
plt.show()
