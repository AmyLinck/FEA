from refactoring.optimizationproblems.continuous_functions import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.basealgorithms.pso import PSO
import random
import time

import functools
# from stat_analysis import factor_graphing
import sys


if __name__ == '__main__':
    dim = 1000
    f = Function(20, shift_data_file="f20_o.txt")
    outputcsv = open('MeetRandom/Experiment_1000_f20_thing.csv', 'a')
    print(f.function_to_call)

    dg_file = open(f'MeetRandom/Graph4_{f.function_to_call}_thing.csv', 'a')

    dim = 1000
    dg_epsilon = 0.001

    k = "epsilon, pop_size, Thing, Thing Fac, Thing Runs, Thing PSO"
    outputcsv.write(k + '\n')
    s = time.time()
    print(dg_epsilon)

    f.evals = 0
    thing = FactorArchitecture(dim=dim)
    thing.load_txt_architecture('F20_arch', 1000)
    thing.save_architecture(f'MeetRandom/{f.function_to_call}_thing')

    for pop_size in [10]:
        for trial in range(25):
            outputcsv.write(f'{dg_epsilon},{pop_size},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_thing")
            print(f"DG {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=dg_file)
            fea_run, pso_runs = fea.run()
            print(f"DG, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)}\n')
            outputcsv.flush()
