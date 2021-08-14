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


def output_callback(output_file, fea, fea_run):
    # print(f'WRITING:  , , {fea_run}, {fea.global_fitness}')
    output_file.write(f" , , {fea_run}, {fea.global_fitness}\n")
    output_file.flush()



if __name__ == '__main__':
    dim = 1000


    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/Experiment_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/Experiment_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        outputcsv = open('MeetRandom/tuning_200_f5.csv', 'a')
        f = Function(3, shift_data_file="f03_o.txt")

    print(f.function_to_call)

    dg_file = open(f'MeetRandom/Graph_{f.function_to_call}_DG', 'a')
    single_file = open(f'MeetRandom/Graph_{f.function_to_call}_Single.csv', 'a')
    odg_file = open(f'MeetRandom/Graph_{f.function_to_call}_ODG.csv', 'a')
    tree_file = open(f'MeetRandom/Graph_{f.function_to_call}_tree.csv', 'a')
    tree2_file = open(f'MeetRandom/Graph_{f.function_to_call}_tree2.csv', 'a')

    random_iteration = [5, 45, 50, 100, 300, 500, 1000, 3000, 5000, 10000]  # 5, 50, 100, 200, 500, 1000, 2000, 5000, 10000

    dim = 1000
    dg_epsilon = 0.001

    k = "epsilon, pop_size, DG, DG Fac, DG Runs, DG PSO, Single, Single Fac, Single Runs, Single PSO, ODG, ODG Fac, ODG Runs, ODG PSO, Tree, Tree fac, Tree Runs, Tree PSO, Tree2, Tree2 fac, Tree2 Runs, Tree2 PSO, PSO"
    outputcsv.write(k + '\n')
    s = time.time()
    print(dg_epsilon)

    f.evals = 0
    odg = FactorArchitecture(dim=dim)
    odg.overlapping_diff_grouping(f, dg_epsilon)
    odg.save_architecture(f'MeetRandom/{f.function_to_call}_odg')

    dg = FactorArchitecture(dim=dim)
    print('starting dg')
    f.evals = 0
    dg.diff_grouping(f, dg_epsilon)
    dg.save_architecture(f'MeetRandom/{f.function_to_call}_dg')

    ccea = FactorArchitecture(dim=dim)
    print('starting single')
    f.evals = 0
    ccea.single_grouping()
    ccea.save_architecture(f'MeetRandom/{f.function_to_call}_single')

    im = RandomTree(f, dim, 3000, dg_epsilon, 0.000001)

    print('starting random')
    T = im.run(5)
    print("finished Random ")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(T)
    meet.save_architecture(f"MeetRandom/{f.function_to_call}_rand")

    meet2 = FactorArchitecture(dim=dim)
    meet2.MEET2(T)
    meet2.save_architecture(f"MeetRandom/{f.function_to_call}_2_rand")

    for pop_size in [10]:
        for trial in range(25):
            outputcsv.write(f'{dg_epsilon},{pop_size},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_dg")
            print(f"DG {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=dg_file)
            fea_run, pso_runs = fea.run()
            print(f"DG, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_single")
            print(f"Single {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=single_file)
            fea_run, pso_runs = fea.run()
            print(f"Single, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_odg")
            print(f"ODG {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=odg_file)
            fea_run, pso_runs = fea.run()
            print(f"ODG, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_rand")
            print(f"Rand {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=tree_file)
            fea_run, pso_runs = fea.run()
            print(f"Rand, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)},')
            outputcsv.flush()

            fa = FactorArchitecture()
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_2_rand")
            print(f"Rand {len(fa.factors)}")

            f.evals = 0
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=tree2_file)
            fea_run, pso_runs = fea.run()
            print(f"Rand, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)},')
            outputcsv.flush()

            f.evals = 0
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
