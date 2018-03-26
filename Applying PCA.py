# =============================================================================
# Applying PCA
# =============================================================================
from copy import copy
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
%matplotlib inline
# Train has been Preprocessed to contain only Continous Variable

X = copy(train)
X = scale(X)


pca = PCA()


pca.fit(X)

var= pca.explained_variance_ratio_




#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print (var1)
plt.plot(var1)

#as it can be easily seen that the graph can easily be explained with the help of 
# least number of Variables so taking 100
pca = PCA(n_components=100)
pca.fit(X)
X1=pca.fit_transform(X)

print X1

