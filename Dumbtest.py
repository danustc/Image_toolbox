"""
Last update: 04/25/2016
Test affine transformation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA


def ICA_test(n_samples = 2000):
    time = np.linspace(0, 10, n_samples)
    s1 = np.sin(2*time) + 1.50
    s2 = np.sign(np.sin(3*time))-0.20
    s3 = signal.sawtooth(2*np.pi*time)

    S = np.c_[s1, s2, s3]
    S+= 0.1*np.random.normal(size = S.shape)


    A = np.random.randn(3,3)
    X = np.dot(S, A.T)
    ica = FastICA(n_components = 3)
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_
    X_ = np.dot(S_, A_.T) + ica.mean_
    assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
    print("The original mixing matrix:\n", A.T)
    print("The calculated mixing matrix:\n", A_.T)
    print(ica.mean_)


    return S, S_, X, X_

if __name__ =='__main__':
    n_samples = 500
    Si, Sf, Xi, Xf = ICA_test(n_samples)
    fig = plt.figure(figsize = (7,7.5))
    ax1 = fig.add_subplot(311)
    ax1.plot(Si)
    ax1.set_title('Original source')
    ax1.set_xticklabels([])
    ax2 = fig.add_subplot(312)
    ax2.plot(Xi)
    ax2.set_title('Mixed signal')
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(313)
    ax3.plot(Sf)
    ax3.set_title('Recoverd signal')
    plt.tight_layout()
    plt.savefig('ica_source')
    plt.close()
