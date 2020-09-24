"""
Topology Generation: This is a VERY VERY confusing topic.

With d = 5, there are 5 variables, x_i => x_0, x_1, x_2, x_3, x_4. This means, in any list,
the variable corresponds to the index. This is a convention and hopefully a space
saving one.

With the "i+1 functions" (Benchmark functions), there are always d - 1 factors.

Of the four sort of topological elements required by the algorithm, there are two kinds:

1. Lists where the indices implicitly refer to swarms and the values of the lists
   refer to variables or swarms.
2. Lists where the indices implicitly refer to variables and the values of the lists
   refer to swarms.

Factors and neighbors fall into the first group. The factor for swarm 0 is [0, 1].
The neighbor of swarm 0 is swarm 1 (because they have variables in common).

Arbiters fall into the second group as do optimizers. The arbiter of variable 0 is swarm 0.
An arbiter of variable 3 is swarm 3 (0-3). Similarly, optimizers are lists of swarms
that optimize for a specific index (variable). Variable 1 is optimized by [0, 1] which is
to say both swarm 0 and swarm 1.

factors = [[0, 1], [1, 2], [2, 3], [3, 4]] # # of factors and swarms
neighbors =   [[1], [0, 2], [1, 3], [2]]
arbiters  =   [0, 1, 2, 3, 3]
optimizers =  [[0], [0, 1], [1, 2], [2, 3], [3]]
"""
import math

from benchmarks import _benchmarks
import numpy as np
from numpy import linalg as la
from copy import copy, deepcopy


def generate_topology(d):
    """
    Generates the topological references needed for these i+1 problems.
    :param d: number of variables
    :return: factors, arbiters, optimizers, neighbors
    """
    factors = zip(range(0, d), range(1, d))
    arbiters = list(range(0, d - 1)) + [d - 2]
    optimizers = []
    for v in range(d):
        optimizer = []
        for i, factor in enumerate(factors):
            if v in factor:
                optimizer.append(i)
        optimizers.append(optimizer)
    neighbors = []
    for i, factor in enumerate(factors):
        neighbor = []
        for j, other_factor in enumerate(factors):
            if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                neighbor.append(j)
        neighbors.append(neighbor)
    return factors, arbiters, optimizers, neighbors


def generate_linear_factors(d, width=2, offset=1):
    """
    calculate how many optimizers we have with the given
    width and offset and make sure the variables in
    the first and last factor have an equal number of optimizers.
    :param d:
    :param width:
    :param offset:
    :return:
    """
    factors = list(zip(*[range(i, d, offset) for i in range(0, width)]))
    return factors


# this might work; it might not work with
# other offsets quite right. Need to figure out the
# math.
# >>> for i in xrange( 4 - 1 + 1):
# ...   print [0, 1, 2, 3][0:-i]
# ...
# []
# [0, 1, 2]
# [0, 1]
# [0]
# >>> for i in xrange( 4 - 1 + 1):
# ...   print [4, 5, 6, 7][i:-1]
# ...
# [4, 5, 6]
# [5, 6]
# [6]
# []
def generate_equal_optimizer_factors(d, width=2, offset=1):
    pass


# def

def nominate_arbiters(factors):
    assignments = {}
    for i, factor in enumerate(factors[:-1]):
        for j in factor:
            if j not in factors[i + 1] and j not in assignments:
                assignments[j] = i
    for j in factors[-1]:
        if j not in assignments:
            assignments[j] = len(factors) - 1
    keys = list(assignments.keys())
    keys.sort()
    arbiters = [assignments[k] for k in keys]
    return arbiters


# def

def calculate_optimizers(d, factors):
    optimizers = []
    for v in range(d):
        optimizer = []
        for i, factor in enumerate(factors):
            if v in factor:
                optimizer.append(i)
        optimizers.append(optimizer)
    return optimizers


# def

def determine_neighbors(factors):
    neighbors = []
    for i, factor in enumerate(factors):
        neighbor = []
        for j, other_factor in enumerate(factors):
            if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                neighbor.append(j)
        neighbors.append(neighbor)
    return neighbors


def generate_linear_topology(d, width=2, offset=1):
    """
    With the default, the topology generated is the same as above but as you
    increase the width, the factors get larger.
    :param d:
    :param width:
    :param offset:
    :return:
    """
    assert offset <= width
    if offset == width:
        print("WARNING - offset and width are equal; the factors will not overlap.")
    factors = generate_linear_factors(d, width, offset)
    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)
    print('linear topology factors: ', factors)
    print(arbiters, optimizers, neighbors)
    return factors, arbiters, optimizers, neighbors


# def

def generate_linear_factors_plus_full_factor(d, width=2, offset=1):
    assert offset <= width
    if offset == width:
        print("WARNING - offset and width are equal; the factors will not overlap.")
    factors = generate_linear_factors(d, width, offset)
    arbiters = nominate_arbiters(factors)
    factors.append(tuple(range(0, d)))
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)
    return factors, arbiters, optimizers, neighbors


# def


def rotate(xs, n):
    return xs[n:] + xs[:n]


def generate_ring_topology(d, width=2):
    arbiters = list(range(0, d))
    factors = zip(*[rotate(arbiters, n) for n in range(0, width)])
    optimizers = []
    for v in range(d):
        optimizer = []
        for i, factor in enumerate(factors):
            if v in factor:
                optimizer.append(i)
        optimizers.append(optimizer)
    neighbors = []
    for i, factor in enumerate(factors):
        neighbor = []
        for j, other_factor in enumerate(factors):
            if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                neighbor.append(j)
        neighbors.append(neighbor)
    print(factors)
    return factors, arbiters, optimizers, neighbors


def generate_duplicated_topology(d, width=2):
    arbiters = list(range(0, d))
    factors = list(zip(*[range(i, d) for i in range(0, width)]))
    last_factor = factors[-1]
    factors = factors + (width - 1) * [last_factor]
    optimizers = []
    for v in range(d):
        optimizer = []
        for i, factor in enumerate(factors):
            if v in factor:
                optimizer.append(i)
        optimizers.append(optimizer)
    neighbors = []
    for i, factor in enumerate(factors):
        neighbor = []
        for j, other_factor in enumerate(factors):
            if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                neighbor.append(j)
        neighbors.append(neighbor)
    return factors, arbiters, optimizers, neighbors


"""
DIFFERENTIAL GROUPING
Omidvar et al. 2010
"""


def generate_overlapping_diff_grouping(_function, d, epsilon):
    """
    Use differential grouping approach to determine factors.
    Package sympy can calculate partial derivatives:
    x, y, z = symbols('x y z', real=True)
    f = 4*x*y + x*sin(z) + x**3 + z**8*y
    diff(f, x)
    4*y + sin(z) + 3*x**2
    :return:
    """

    size = deepcopy(d)
    dimensions = np.arange(start=0, stop=d)
    lbound = -100
    ubound = 100
    factors = []
    separate_variables = []
    function_evaluations = 0
    loop = 0

    for i, dim in enumerate(dimensions):
        # initialize for current iteration
        print('********************************* NEW LOOP ****************************')
        print("current dimension = ", dim)
        curr_factor = [dim]

        p1 = np.multiply(lbound, np.ones(d))
        p2 = np.multiply(lbound, np.ones(d))  # python does weird things if you set p2 = p1
        p2[i] = ubound
        delta1 = _function(p1) - _function(p2)
        function_evaluations += 2

        for j in range(i + 1, size):
            p3 = deepcopy(p1)
            p4 = deepcopy(p2)
            p3[dimensions[j]] = 0  # grabs dimension to compare to, same as index
            p4[dimensions[j]] = 0

            delta2 = _function(p3) - _function(p4)
            function_evaluations += 2

            if abs(delta1 - delta2) > epsilon:
                curr_factor.append(dimensions[j])
            # else:
            #     print('absolute difference: ', abs(delta1 - delta2), 'vs epsilon: ', epsilon)

        if len(curr_factor) == 1:
            separate_variables.extend(curr_factor)
        else:
            factors.append(tuple(curr_factor))

        loop += 1

    factors.append(tuple(separate_variables))
    print(factors)

    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)

    return factors, arbiters, optimizers, neighbors


def generate_diff_grouping(_function, d, epsilon, m=0):
    """

    :param _function:
    :param d:
    :param epsilon:
    :return:
    """
    size = deepcopy(d)
    dimensions = np.arange(start=0, stop=d)
    curr_dim_idx = 0
    lbound = -100
    ubound = 100
    factors = []
    separate_variables = []
    function_evaluations = 0
    loop = 0

    while size > 0:
        # initialize for current iteration
        # print('********************************* NEW LOOP ****************************')
        # print("dimensions in loop ", loop, ": ", dimensions)
        curr_factor = [dimensions[0]]

        p1 = np.multiply(lbound, np.ones(d))
        p2 = np.multiply(lbound, np.ones(d))  # python does weird things if you set p2 = p1
        p2[curr_dim_idx] = ubound
        if m == 0:
            delta1 = _function(p1) - _function(p2)
        else:
            delta1 = _function(p1, m_group = m) - _function(p2, m_group = m)
        function_evaluations += 2

        for j in range(1, size):
            p3 = deepcopy(p1)
            p4 = deepcopy(p2)
            p3[dimensions[j]] = 0  # grabs dimension to compare to, corresponds to python index
            p4[dimensions[j]] = 0

            if m == 0:
                delta2 = _function(p3) - _function(p4)
            else:
                delta2 = _function(p3, m_group = m) - _function(p4, m_group = m)
            function_evaluations += 2

            if abs(delta1 - delta2) > epsilon:
                curr_factor.append(dimensions[j])
            # else:
            #     print('absolute difference: ', abs(delta1 - delta2), 'vs epsilon: ', epsilon)

        if len(curr_factor) == 1:
            separate_variables.extend(curr_factor)
            # dimensions = np.delete(dimensions, 0)
        else:
            factors.append(tuple(curr_factor))

        # Final adjustments
        indeces_to_delete = np.searchsorted(dimensions, curr_factor)
        dimensions = np.delete(dimensions, indeces_to_delete)  # remove j from dimensions
        size = len(dimensions)
        if size != 0:
            curr_dim_idx = dimensions[0]

        loop += 1
    if len(separate_variables) != 0:
        factors.append(tuple(separate_variables))
    print(factors)

    arbiters = nominate_arbiters(factors)
    optimizers = calculate_optimizers(d, factors)
    neighbors = determine_neighbors(factors)

    return factors, arbiters, optimizers, neighbors, separate_variables

"""
FUZZY CLUSTERING GROUPING
"""

"""
based on variable interaction matrix, create fuzzy hierarchical clustering

Hierarchical clustering based on similarity matrix, joining from bottom using linkage strategy, 
what if we start linking by looking at at pariwise variable interaction (eg bayes net: mutual information, statistical data: correlation). 
Take whatever measure you're using and create variable interaction matrix -> creates symmetric matrix (needs to be positive definite!), 
treat it like a a similarity matrix in HAC, build dendrogram based on matrix, effectively creating clusters. Scott developed FUZZY spectral HAC algorithm. 
Eigen decomposition of variable interaction matrix, fuzzy c-means creates overlapping factors. 
Doesn't have to be hierarchical, but might create less factors and larger clusters 

Based on concept of Laplacian matrix; but instead of degree and adjacancy matrix, user interaction and total interaction between variables.  
"""

"""
Need a graph. 

Clustering
    - A = adjacency matrix of graph
    - D = diagonal matrix: d_i,i = \sum_j a_i,j
    - L = Laplacian = L = D^-1/2 A D^-1/2
    
    - Spectral decomp of L, use k-largest eiganvectors (sorted by eiganvalue)
    - k means cluster to the eiganvectors of L
    - use fuzzy to assign to multiple clusters
"""


# A is the adjacency matrix
# data is the list of vectors (similarity matrix?)
# k is number of eiganvectors to use
# ep is epsilon for c-means
def fuzzy_clustering(A, data, k, ep):
    D = create_diagonal(A)
    D_inv = la.inv(D)
    prod1 = np.matmul(D_inv, A)
    L = np.matmul(prod1, D_inv)  # creates the laplacian
    w, v = la.eig(L)  # Computes eigenvalues and eigenvectors of L
    v = np.transpose(v)  # Changes col vectors into row vectors so easier to grab
    sorted_pairs = sorted(zip(w, v), reverse=True)

    tuples = zip(*sorted_pairs)
    w, v = [list(tup) for tup in tuples]  # The sorted eigenvalues, eigenvectors

    eigenvectors = v[:k]

    # Make the vectors positive
    # TODO: check if it matters which components are negative (flip sign if most are negative?)
    for i in range(len(eigenvectors)):  # len(eigenvectors) = k
        if eigenvectors[i][0] < 0:  # if the first component is negative make it positive
            eigenvectors[i] = -1 * eigenvectors[i]

    # Now use c means to cluster next to first k elements in v
    # I assume that the eigenvectors are the centroids
    max_magnitude = max(la.norm(x) for x in data)
    normalized = data/max_magnitude
    distances = []  # store distance from each point to each centroid

    # Compute the distances to clusters
    for point in normalized:
        # point is a vector
        d = [la.norm(point - v) for v in eigenvectors]  # gets the distance to each eigenvector
        distances.append(d)

    clusters = [[] for i in range(len(eigenvectors))]  # empty 2d array
    # Assign to clusters
    for i in range(len(distances)):
        # i is the variable
        for d in range(len(distances[i])):
            if distances[i][d] < ep:  # if distance to eigenvector is less than threshold, add the variable to cluster
                clusters[d].append(i)

    return clusters


def create_diagonal(A):
    D = np.zeros(shape=A.shape)
    width = A.shape[0]
    for i in range(width):  # assume square

        # Since D is diagonal (and every element is positive) then D^1/2 is the square root of every element in D
        # D^1/2_i,i = sqrt(D_i,i)

        # D[i][i] = sum(A[i])
        D[i][i] = math.sqrt(sum(A[i]))

    return D
