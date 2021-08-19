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

    f = f20
    method = 'odg'

    fa = FactorArchitecture()
    fa.load_architecture(f"MeetRandom/{f.function_to_call}_{method}")

    print(len(fa.factors))
    total = 0
    for fac in fa.factors:
        total += len(fac)
    print(total / len(fa.factors))
    print(fac)


