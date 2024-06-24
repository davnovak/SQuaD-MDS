import numpy as np
from sklearn.decomposition import PCA
from .optimisers import SQuaD_MDS, SQuaD_MDS_tsne

class SQuadMDS:
    """SQuad-MDS
    """

    def __init__(
        self,
        n_components=2,
        metric='euclidean', # or 'relative rbf distance'
        n_iter=1000,
        learning_rate=550,
        exaggerate_distances=True,
        stop_exaggeration=0.6,
        tsne_hybrid=False,
        tsne_lr_multiplier=5.,
        PP=5.,
        tsne_exaggeration=2.,
        random_state=None
    ):
        self.hparams = {
            'metric': metric,
            'n iter': n_iter,
            'LR': learning_rate,
            'exaggerate D': exaggerate_distances,
            'stop exaggeration': stop_exaggeration,
            'PP': PP,
            'tsne exa': tsne_exaggeration,
            'tsne LR multiplier': tsne_lr_multiplier
        }
        self.n_components = n_components
        self.random_state = random_state
        self.tsne_hybrid = tsne_hybrid

    def fit_transform(self, X):
        if self.random_state is not None:
            np.random.seed(seed=self.random_state)
        Xld = PCA(n_components=self.n_components, whiten=True, copy=True, random_state=self.random_state).fit_transform(X).astype(np.float64)
        Xld *= 10/np.std(Xld)

        if not self.tsne_hybrid:
            SQuaD_MDS(self.hparams, X, Xld)
        else:
            SQuaD_MDS_tsne(self.hparams, X, Xld)
        return Xld
    