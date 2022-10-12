from datetime import timedelta

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.factory import get_problem, get_reference_directions

from MOO.MOEA import NSGA2, SPEA2, MOEAD, MOEA
from FEA.factorarchitecture import FactorArchitecture
from MOO.MOFEA import MOFEA
from utilities.util import *

import os, re, time, pickle
from pymoo.problems.many.dtlz import DTLZ1

dimensions = 1000
sizes = [100, 100, 200, 200]  # , 200, 100]
overlaps = [80, 100, 160, 200]  # 100, 160, 80]  # , 10, 20]
fea_runs = [20]
ga_run = 20
population = 500
nr_obj = 3

ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=12)
pf = get_problem("dtlz1", n_var=dimensions, n_obj=nr_obj).pareto_front(ref_dirs)
reference_point = np.max(pf, axis=0)
FA = FactorArchitecture(dimensions)

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

moea1 = partial(SPEA2, population_size=population, ea_runs=ga_run)
moea2 = partial(NSGA2, population_size=population, ea_runs=ga_run)
moea3 = partial(MOEAD, ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff())

partial_methods = [moea1, moea2, moea3]
names=['SPEA2', 'NSGA2', 'MOEAD']

@add_method(MOEA)
def calc_fitness(variables, gs=None, factor=None):
    if gs is not None and factor is not None:
        full_solution = [x for x in gs.variables]
        for i, x in zip(factor, variables):
            full_solution[i] = x
    else:
        full_solution = variables
    dtlz = get_problem("dtlz1", n_var=dimensions, n_obj=nr_obj)
    objective_values = dtlz.evaluate(full_solution)
    return tuple(objective_values)


for s, o in zip(sizes, overlaps):
    FA.linear_grouping(s, o)
    FA.get_factor_topology_elements()
    FA.method = "linear_1_1"
    for j,alg in enumerate(partial_methods):
        if s == o:
            name = 'CC' + names[j]
        else:
            name = 'F' + names[j]
        for i in range(5):
            print('##############################################\n', i)
            for fea_run in fea_runs:
                start = time.time()
                filename = path + '/results/DTLZ1/' + name + '/' + name + '_DTLZ1_' + str(nr_obj) + \
                        '_objectives_fea_runs_' + str(fea_run) + '_grouping_' + str(s) + '_' + \
                        str(o) + time.strftime('_%d%m%H%M%S') + '.pickle'
                feamoo = MOFEA(fea_run, factor_architecture=FA, base_alg=alg, dimensions=dimensions,
                            value_range=[0.0, 1.0], ref_point=reference_point)
                feamoo.run()
                end = time.time()
                try:
                    file = open(filename, "wb")
                except FileNotFoundError:
                    os.path.mkdir(path + '/results/DTLZ1/' + name + '/')
                pickle.dump(feamoo, file)
                elapsed = end - start
                print(
                    "FEA with ga runs %d and population %d %d took %s" % (fea_run, s, o, str(timedelta(seconds=elapsed))))
