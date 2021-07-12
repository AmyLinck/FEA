from refactoring.optimizationProblems.continuous_functions import Function
from refactoring.utilities.varinteraction import MEE, RandomTree
from refactoring.FEA.factorevolution import FEA
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.baseAlgorithms.pso import PSO

# from stat_analysis import factor_graphing
import sys

if __name__ == '__main__':


    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/1000_dim_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/1000_dim_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        outputcsv = open('./MeetRandom/1000_dim_f5.csv', 'a')
        f = Function(5, shift_data_file="f05_op.txt", matrix_data_file="f05_m.txt")
    print(f.function_to_call)

    dim = 1000
    # outputfile.write("Dim: " + str(dim) + " Seed = 1 \t ODG and DG\n")
    random_iteration = [5, 45, 50, 100, 300, 500, 1000, 3000, 5000]  # 5, 50, 100, 200, 500, 1000, 2000, 5000, 10000
    k = "DG,ODG,MEET," + ','.join([f'Rand {sum(random_iteration[0:i + 1])}' for i in range(len(random_iteration))])
    outputcsv.write(k + '\n')

    # odg = FactorArchitecture(dim=dim)
    # print('starting odg')
    # odg.overlapping_diff_grouping(f, 0.001)
    # odg.save_architecture('MeetRandom/odg2')
    #
    # dg = FactorArchitecture(dim=dim)
    # print('starting dg')
    # dg.diff_grouping(f, 0.001)
    # dg.save_architecture('MeetRandom/dg2')

    for i in range(15):
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
        # meet.save_architecture("MeetRandom/meet2")

        # factor_graphing(meet.factors, "./MeetRandom/imgs/meet/")

        fa = FactorArchitecture()
        print("FEA MEET")
        fa.load_architecture("MeetRandom/meet")
        fea = FEA(f, 10, 10, 3, fa, PSO, seed=i)
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
            fea = FEA(f, 10, 10, 3, fa, PSO, seed=i)
            fea.run()
            outputfile.write(f"Rand {total}, \t{fea.global_fitness}\n")
            print(fea.global_fitness)

            summary[f'Rand {total}'] = fea.global_fitness

        fa = FactorArchitecture()
        print("FEA ODG")
        fa.load_architecture("MeetRandom/odg")
        fea = FEA(f, 10, 10, 3, fa, PSO, seed=i)
        fea.run()
        outputfile.write(f"ODG, \t\t{fea.global_fitness}\n")
        print(fea.global_fitness)
        summary['ODG'] = fea.global_fitness

        fa = FactorArchitecture()
        print("FEA DG")
        fa.load_architecture("MeetRandom/dg")
        fea = FEA(f, 10, 10, 3, fa, PSO, seed=i)
        fea.run()
        outputfile.write(f"DG, \t\t{fea.global_fitness}\n")
        print(fea.global_fitness)
        summary['DG'] = fea.global_fitness

        print(summary)

        keys = k.split(',')
        if all(elem in keys for elem in summary.keys()):
            print("writing to file")
            line_out = ','.join([str(summary[key]) for key in keys])
            print(line_out)
            outputcsv.write(line_out + '\n')
        else:
            print(f'{summary.keys()} != {keys}')
        outputfile.close()
