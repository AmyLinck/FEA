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
    f3 = Function(3, shift_data_file='f03_o.txt')
    f5 = Function(5, shift_data_file='f05_op.txt', matrix_data_file='f05_m.txt')
    f11 = Function(11, shift_data_file='f11_op.txt', matrix_data_file='f11_m.txt')
    f17 = Function(17, shift_data_file='f17_op.txt')
    f20 = Function(19, shift_data_file='f20_o.txt')

    functions = [f3, f5, f11, f17, f20]
    files = ['f03.txt', 'f05.txt', 'f11.txt', 'f17.txt', 'f20.txt']

    dg_epsilon = 0.001
    dim = 1000

    for i, f in enumerate(functions):
        # f.evals = 0
        # odg = FactorArchitecture(dim=dim)
        # odg.overlapping_diff_grouping(f, dg_epsilon)
        # odg.save_architecture(f'MeetRandom/{f.function_to_call}_odg')

        dg = FactorArchitecture(dim=dim)
        print('starting dg')
        f.evals = 0
        dg.diff_grouping(f, dg_epsilon)
        dg.save_architecture(f'MeetRandom/{f.function_to_call}_dg')
        dg.get_factor_topology_elements()

        f.evals = 0
        fea = FEA(f, 50, 15, 10, dg, PSO, seed=1)
        fea_run, pso_runs = fea.run()
        print(f"DG, \t\t{len(fa.factors)}, {fea.global_fitness}\n")

        continue

        fa = FactorArchitecture(dim = 1000)
        fa.load_txt_architecture(files[i])
        print(len(fa.factors))


