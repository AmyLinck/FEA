from refactoring.optimizationproblems.continuous_functions import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.basealgorithms.pso import PSO

import time

import functools
# from stat_analysis import factor_graphing
import sys


def output_callback(output_file, fea, fea_run):
    # print(f'WRITING:  , , {fea_run}, {fea.global_fitness}')
    output_file.write(f" , , {fea_run}, {fea.global_fitness}\n")
    output_file.flush()


if __name__ == '__main__':
    dim = 1000

    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/Tune_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/Tune_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        outputcsv = open('MeetRandom/tuning_200_f5.csv', 'a')
        f = Function(3, shift_data_file="f03_o.txt")
    print(f.function_to_call)

    random_iteration = [5, 45, 50, 100, 300, 500, 1000, 3000, 5000, 10000]  # 5, 50, 100, 200, 500, 1000, 2000, 5000, 10000

    dim = 1000
    dg_epsilon = 0.001

    k = "epsilon, pop_size, MEET, MEET Fac, MEET Runs, MEET PSO, PSO"
    outputcsv.write(k + '\n')
    s = time.time()
    print(dg_epsilon)
    f.evals = 0
    print("Starting MEET IM")
    im = MEE(f, dim, 3000, 0, 0.001, 0.000001, use_mic_value=True)
    IM = im.get_IM()
    print("finished IM")
    f.evals = 0
    meet = FactorArchitecture(dim=dim)
    meet.MEET(IM)
    print("finished MEET")
    meet.save_architecture(f"MeetRandom/{f.function_to_call}_meet")

    for pop_size in [50, 25, 100]:
        for trial in range(25):
            outputcsv.write(f'{dg_epsilon},{pop_size},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_meet")
            print(f"DG {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial)
            fea_run, pso_runs = fea.run()
            print(f"MEET, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs)/len(pso_runs)},')
            outputcsv.flush()

            pso = PSO(generations=3000, population_size=1000, function=f, dim=dim)  # generations=3000
            gbest = pso.run()
            outputcsv.write(f'{pso.gbest.fitness}\n')


        # print("PSO")
        # pso = PSO(generations=3000, population_size=pop_size, function=f, dim=dim)  # generations=3000
        # gbest = pso.run()
        # summary['PSO'] = pso.gbest.fitness
        #
        # print(summary)
        #
        # keys = k.split(',')
        # if all(elem in keys for elem in summary.keys()):
        #     line_out = ','.join([str(summary[key]) for key in keys])
        #     outputcsv.write(line_out + '\n')
        #     outputcsv.flush()
        # else:
        #     print(f'{summary.keys()} != {keys}')
        # outputfile.close()
