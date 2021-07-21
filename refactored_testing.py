from refactoring.optimizationproblems.continuous_functions import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.basealgorithms.pso import PSO

import functools
# from stat_analysis import factor_graphing
import sys


def output_callback(output_file, fea, fea_run):
    # print(f'WRITING:  , , {fea_run}, {fea.global_fitness}')
    output_file.write(f" , , {fea_run}, {fea.global_fitness}\n")
    output_file.flush()


if __name__ == '__main__':
    dim = 200

    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/tuning_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/tuning_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        outputcsv = open('MeetRandom/tuning_200_f5.csv', 'a')
        f = Function(5, shift_data_file="f05_op.txt", matrix_data_file="f05_m.txt")
    print(f.function_to_call)


    # outputfile.write("Dim: " + str(dim) + " Seed = 1 \t ODG and DG\n")
    k = "Pop, PSO,FEA_ITER, ODG"
    outputcsv.write(k + '\n')

    odg = FactorArchitecture(dim=dim)
    print('starting odg')
    odg.overlapping_diff_grouping(f, 0.001)
    odg.save_architecture('MeetRandom/architectures/odg')

    dg = FactorArchitecture(dim=dim)
    print('starting dg')
    dg.diff_grouping(f, 0.001)
    dg.save_architecture('MeetRandom/architectures/dg')


    callback_func = functools.partial(output_callback, outputcsv)
    pso_options = [10, 15, 20]
    pop_size = [50, 100, 300, 500]
    fea_max = 25
    for i in range(3):
        for pop in pop_size:
            for pso_iter in pso_options:
                print(f"I: {i}\tpop: {pop}\tPSO_ITER: {pso_iter}")
                outputcsv.write(f"{pop}, {pso_iter}, , \n")

                outputfile = open('./MeetRandom/tuning.txt', 'a')
                print("running")

                summary = {}
                outputfile.write("Dim: " + str(dim) + " Random Init\n")

                fa = FactorArchitecture()
                print("FEA ODG")
                fa.load_architecture("MeetRandom/architectures/odg")
                fea = FEA(f, fea_max, pso_iter, pop, fa, PSO, seed=i, callback=callback_func)
                fea.run()
                outputfile.write(f"ODG, \t\t{fea.global_fitness}\n")
                print(fea.global_fitness)
                summary['ODG'] = fea.global_fitness

                print(summary)

                # keys = k.split(',')
                # if all(elem in keys for elem in summary.keys()):
                #     print("writing to file")
                #     line_out = ','.join([str(summary[key]) for key in keys])
                #     print(line_out)
                #     outputcsv.write(line_out + '\n')
                # else:
                #     print(f'{summary.keys()} != {keys}')
                outputfile.close()
