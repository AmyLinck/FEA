import numpy as np
from topology import *

adj = np.array([[1,1/3,1/9],[1/3,1,1/2],[1/9,1/2,1]])
data = np.array([[1,3,9],[4,3,2],[7,5,6]])
print(fuzzy_clustering(adj, data, 2, 1))


