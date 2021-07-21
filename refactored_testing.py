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
    dim = 1000

    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/Trial_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/Trial_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        outputcsv = open('MeetRandom/tuning_200_f5.csv', 'a')
        f = Function(5, shift_data_file="f05_op.txt", matrix_data_file="f05_m.txt")
    print(f.function_to_call)


    # outputfile.write("Dim: " + str(dim) + " Seed = 1 \t ODG and DG\n")
    dim = 1000
    # outputfile.write("Dim: " + str(dim) + " Seed = 1 \t ODG and DG\n")
    random_iteration = [5, 45, 50, 100, 300, 500, 1000, 3000, 5000, 10000]  # 5, 50, 100, 200, 500, 1000, 2000, 5000, 10000
    k = "PSO,DG,ODG,MEET," + ','.join([f'Rand {sum(random_iteration[0:i + 1])}' for i in range(len(random_iteration))])
    outputcsv.write(k + '\n')

    odg = FactorArchitecture(dim=dim)
    print('starting odg')
    odg.overlapping_diff_grouping(f, 0.001)
    odg.save_architecture('MeetRandom/architectures/odg.arch')

    dg = FactorArchitecture(dim=dim)
    print('starting dg')
    dg.diff_grouping(f, 0.001)
    dg.save_architecture('MeetRandom/architectures/dg.arch')

    print("Starting MEET IM")
    im = MEE(f, dim, 3000, 0, 0.001, 0.000001, use_mic_value=True)
    IM = im.get_IM()
    print("finished IM")
    meet = FactorArchitecture(dim=dim)
    meet.MEET(IM)
    print("finished MEET")
    meet.save_architecture("MeetRandom/meet.arch")

    fea_iter = 25
    pop_size = 1000
    func_evals = 3000000
    for i in range(25):
        outputfile = open('./MeetRandom/trial.txt', 'a')
        print("running")

        summary = {}
        outputfile.write("Dim: " + str(dim) + " Random Init\n")

        total = 0
        im = RandomTree(f, dim, 3000, 0.001, 0.000001)
        for it in random_iteration:
            total += it

            print("Starting Random " + str(total))
            T = im.run(it)
            print("finished Random " + str(total))
            meet = FactorArchitecture(dim=dim)
            meet.MEET(T)
            print("finished Random " + str(total))
            meet.save_architecture("MeetRandom/rand" + str(total))

            # factor_graphing(meet.factors, f"./MeetRandom/imgs/rand{total}/")

        # Skip MEET for time
        # print("Starting MEET IM")
        # im = MEE(f, dim, 3000, 0, 0.001, 0.000001, use_mic_value=True)
        # IM = im.get_IM()
        # print("finished IM")
        # meet = FactorArchitecture(dim=dim)
        # meet.MEET(IM)
        # print("finished MEET")
        # meet.save_architecture("MeetRandom/meet")

        # factor_graphing(meet.factors, "./MeetRandom/imgs/meet/")

        fa = FactorArchitecture()
        print("FEA MEET")
        fa.load_architecture("MeetRandom/meet.arch")
        fea = FEA(f, fea_iter, int(func_evals / (fea_iter * pop_size)), pop_size, fa, PSO, seed=i)
        fea.run()
        outputfile.write(f"MEET, \t\t{fea.global_fitness}\n")
        print(fea.global_fitness)
        summary['MEET'] = fea.global_fitness

        total = 0
        for it in random_iteration:
            total += it
            fa = FactorArchitecture()
            print("FEA Rand " + str(total))
            fa.load_architecture("MeetRandom/rand" + str(total))
            fea = FEA(f, fea_iter, int(func_evals / (fea_iter * pop_size)), pop_size, fa, PSO, seed=i)  # pso iter = 120
            fea.run()
            outputfile.write(f"Rand {total}, \t{fea.global_fitness}\n")
            print(fea.global_fitness)

            summary[f'Rand {total}'] = fea.global_fitness

        fa = FactorArchitecture()
        print("FEA ODG")
        fa.load_architecture("MeetRandom/odg.arch")
        fea = FEA(f, fea_iter, int(func_evals / (fea_iter * pop_size)), pop_size, fa, PSO, seed=i)
        fea.run()
        outputfile.write(f"ODG, \t\t{fea.global_fitness}\n")
        print(fea.global_fitness)
        summary['ODG'] = fea.global_fitness

        fa = FactorArchitecture()
        print("FEA DG")
        fa.load_architecture("MeetRandom/dg.arch")
        fea = FEA(f, fea_iter, int(func_evals / (fea_iter * pop_size)), pop_size, fa, PSO, seed=i)
        fea.run()
        outputfile.write(f"DG, \t\t{fea.global_fitness}\n")
        print(fea.global_fitness)
        summary['DG'] = fea.global_fitness

        print("PSO")
        pso = PSO(generations=int(func_evals / pop_size), population_size=pop_size, function=f, dim=dim)  # generations=3000
        gbest = pso.run()
        summary['PSO'] = pso.gbest.fitness

        print(summary)

        keys = k.split(',')
        if all(elem in keys for elem in summary.keys()):
            line_out = ','.join([str(summary[key]) for key in keys])
            outputcsv.write(line_out + '\n')
            outputcsv.flush()
        else:
            print(f'{summary.keys()} != {keys}')
        outputfile.close()
