import numpy as np
from minepy import MINE
from networkx import from_numpy_array, maximum_spanning_tree, connected_components
from random import choice


class MEE(object):
    def __init__(self, func, dim, samples, mic_thresh, de_thresh, delta, use_mic_value=True):
        self.f = func
        self.d = dim
        self.ub = np.ones(self.d) * func.ubound
        self.lb = np.ones(self.d) * func.lbound
        self.samples = samples
        self.mic_thresh = mic_thresh  # mic threshold
        self.de_thresh = de_thresh  # diff equation (de) threshold
        self.delta = delta  # account for small variations
        self.IM = np.zeros((self.d, self.d))
        self.use_mic_value = use_mic_value

        # Define measure
        self.measure = Entropic(self.f, self.d, self.lb, self.ub, self.samples, self.delta, self.de_thresh)

    def get_IM(self):
        self.direct_IM()
        if not self.use_mic_value:
            self.strongly_connected_comps()
        return self.IM

    def direct_IM(self):

        """
        Calculates the Direct Interaction Matrix based on MIC
        :return: direct_IM
        """
        f, dim, lb, ub, sample_size, delta = self.f, self.d, self.lb, self.ub, self.samples, self.delta
        # for each dimension
        for i in range(dim):
            # compare to consecutive variable (/dimension)
            for j in range(i + 1, dim):

                mic = self.measure.compute(i, j)

                if self.use_mic_value:
                    # print(mic, end=", ")
                    self.IM[i, j] = mic
                elif not self.use_mic_value and mic > self.mic_thresh:  # threshold <--------
                    self.IM[i, j] = 1
                    self.IM[j, i] = 1

    def strongly_connected_comps(self):
        from networkx import to_networkx_graph, DiGraph
        from networkx.algorithms.components import strongly_connected_components

        """
        Sets strongly connected components in the Interaction Matrix
        """
        IM_graph = to_networkx_graph(self.IM, create_using= DiGraph)
        strongly_connected_components = strongly_connected_components(IM_graph)
        for component in strongly_connected_components:
            component = list(component)
            for i in range(len(component)):
                for j in range(i + 1, len(component)):
                    self.IM[component[i], component[j]] = 1
                    self.IM[component[j], component[i]] = 1


class RandomTree(object):
    def __init__(self, func, dim, samples, de_thresh, delta):
        self.f = func
        self.d = dim
        self.ub = np.ones(self.d) * func.ubound
        self.lb = np.ones(self.d) * func.lbound
        self.delta = delta  # account for small variations
        r = np.random.random(size=(self.d, self.d))

        self.IM = (r + r.T) - 2  # init IM to be symmetric with random values between [-2, 0]
        # self.IM = np.ones((self.d, self.d)) * -1  # init IM to bunch of -1's (so we can initialize a tree)

        self.samples = samples
        self.de_thresh = de_thresh

        self.iteration_ctr = 0

        # Init tree and graph
        self.G = from_numpy_array(self.IM)  # We don't technically need this in self, but might as well have it
        self.T = maximum_spanning_tree(self.G)  # just make a tree (they're all -1 so it is a boring tree)

        # Define measure
        self.measure = Entropic(self.f, self.d, self.lb, self.ub, self.samples, self.delta, self.de_thresh)

    def run(self, trials):
        """
        Runs a greedy improvement algorithm on the existing tree
        Replaces the cheapest edge, and adds a new edge if it is less expensive then the queried edge
        :param trials: The number of iterations to run
        :return:
        """
        summary = ""
        for i in range(trials):
            self.iteration_ctr += 1  # keep track of global counter to allow for multiple, sequential run calls
            # print("Iteration " + str(self.iteration_ctr))

            edges = list(self.T.edges(data="weight"))
            remove = choice(edges)  # remove a random edge
            # remove = min(edges, key=lambda e: e[2])  # find the cheapest edge
            self.T.remove_edge(remove[0], remove[1])  # delete the edge

            comp1, comp2 = connected_components(self.T)

            node1 = choice(list(comp1))  # generate random start node
            node2 = choice(list(comp2))  # generate random end node

            interact = self.compute_interaction(node1, node2)
            summary += f"\t|\t{remove[2]} --> {interact} "
            if interact > remove[2]:  # if the new random edge is more expensive then the previous one, add it
                self.T.add_edge(node1, node2, weight=interact)
                summary += "Accepted"
            else:  # otherwise add the original one back
                self.T.add_edge(remove[0], remove[1], weight=remove[2])
                summary += "Rejected"
        print(summary)
        return self.T

    def compute_interaction(self, i, j):
        """
        Computes the interaction using MEE between vars i, j
        :param i:
        :param j:
        :return: MIC value
        """
        if self.IM[i][j] > 0:
            return self.IM[i][j]

        mic = self.measure.compute(i, j)

        self.IM[i, j] = mic
        self.IM[j, i] = mic
        return mic


class Measure(object):
    """
    Base class
    """

    def compute(self, i, j):
        return 0


class Entropic(Measure):
    def __init__(self, f, d, lb, ub, samples, delta, de_thresh):
        """
        Uses MEE to compute interaction
        :param f: function
        :param d: dimensions
        :param lb: lower bound matrix
        :param ub: upper bound matrix
        :param samples: number of samples to take
        :param delta: pertubation
        :param de_thresh: threshold value
        """
        self.de_thresh = de_thresh
        self.delta = delta
        self.samples = samples
        self.ub = ub
        self.lb = lb
        self.d = d
        self.f = f

    def compute(self, i, j):
        # number of values to calculate == sample size
        f, dim, lb, ub, sample_size, delta = self.f, self.d, self.lb, self.ub, self.samples, self.delta
        de = np.zeros(sample_size)
        # generate n values (i.e. samples) for j-th dimension
        x_j = np.random.rand(sample_size) * (ub[j] - lb[j]) + lb[j]
        # randomly generate solution -- initialization of function variables
        x = np.random.uniform(lb, ub, size=dim)
        for k in range(1, sample_size):
            cp = x[j]
            x[j] = x_j[k]  # set jth value to random sample value
            y_1 = f.run(x)
            x[i] = x[i] + delta
            y_2 = f.run(x)
            de[k] = (y_2 - y_1) / delta
            # Reset the changes
            x[j] = cp
            x[i] = x[i] - delta

        avg_de = np.mean(de)
        de[de < self.de_thresh] = avg_de  # use np fancy indexing to replace values

        mine = MINE()
        mine.compute_score(de, x_j)
        mic = mine.mic()
        return mic


class DGInteraction(Measure):
    def __init__(self, func, dim, epsilon, lbound, ubound, m=0):
        self.f = func
        self.dim = dim
        self.eps = epsilon
        self.m = m
        self.lbound = lbound
        self.ubound = ubound

    def compute(self, i, j):
        p1 = np.multiply(self.lbound, np.ones(self.dim))  # python does weird things if you set p2 = p1
        p2 = np.multiply(self.lbound, np.ones(self.dim))  # python does weird things if you set p2 = p1
        p2[i] = self.ubound
        if self.m == 0:
            delta1 = self.f.run(p1) - self.f.run(p2)
        else:
            delta1 = self.f.run(p1, m_group=self.m) - self.f.run(p2, m_group=self.m)

        p3 = np.multiply(self.lbound, np.ones(self.dim))
        p4 = np.multiply(self.lbound, np.ones(self.dim))
        p4[i] = self.ubound
        p3[j] = 0  # In factorarcitecture.check_delta it is self.dimensions[j]. In ODG this is equivalent
        p4[j] = 0  # grabs dimension to compare to, same as home

        if self.m == 0:
            delta2 = self.f.run(p3) - self.f.run(p4)
        else:
            delta2 = self.f.run(p3, m_group=self.m) - self.f.run(p4, m_group=self.m)

        return abs(delta1 - delta2)


if __name__ == '__main__':
    from refactoring.optimizationProblems.function import Function
    f = Function(function_number=1, shift_data_file="f01_o.txt")
    mee = MEE(f, 5, 5, 0.1, 0.0001, 0.000001)
    mee.get_IM()