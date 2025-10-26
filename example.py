
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
import myimport

from deslib.dcs import LCA
from deslib.des import KNORAE
from discr_idx import DISi


rng = np.random.RandomState(42)
X, y = make_classification(n_samples=1000, random_state=rng)
xtmp, xte, ytmp, ytrue = train_test_split(X, y, test_size=0.25, random_state=rng)
xtr, xdsel, ytr, ydsel = train_test_split(xtmp, ytmp, test_size=0.33, random_state=rng)

estimator = Perceptron(max_iter=100)
pool_classifiers = BaggingClassifier(estimator, n_estimators=100,random_state=rng).fit(xtr, ytr)

lca = LCA(pool_classifiers, random_state=rng).fit(xdsel, ydsel)
kne = KNORAE(pool_classifiers, random_state=rng).fit(xdsel, ydsel)
disi = DISi(pool_classifiers, random_state=rng).fit(xdsel, ydsel)

print('--- result ---')
print('lca --> ', lca.score(xte, ytrue))
print('kne -->', kne.score(xte, ytrue))
print('disi ->', disi.score(xte, ytrue))

