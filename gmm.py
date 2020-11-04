import pandas as pd
from sklearn.mixture import GaussianMixture

data = pd.read_csv('scaled.csv')

gmm = GaussianMixture(16)
gmm.fit(data)
print(d, '::', gmm.score(data))
