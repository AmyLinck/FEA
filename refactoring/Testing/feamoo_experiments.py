import csv
import os
import pickle
from datetime import timedelta
import time

from refactoring.optimizationproblems.prescription import Prescription
from refactoring.MOO.FEAMOO import FEAMOO
from refactoring.MOO.paretofront import *
from refactoring.basealgorithms.MOO_GA import NSGA2
from refactoring.FEA.factorarchitecture import FactorArchitecture
from refactoring.utilities.field.field_creation import Field

field_names = ['Henrys'] #  'Henrys', 'Sec35West']
current_working_dir = os.getcwd()
print(current_working_dir)
field_1 = pickle.load(open(current_working_dir + '/refactoring/utilities/saved_fields/Henrys.pickle', 'rb')) # /home/alinck/FEA
field_2 = pickle.load(open('/home/alinck/FEA/refactoring/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open('/home/alinck/FEA/refactoring/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_1]

fea_runs = 100
ga_runs = [100]
population_sizes= [500]

for i,field in enumerate(fields_to_test):
    FA = FactorArchitecture(len(field.cell_list))
    # FA.linear_grouping(10, 5)
    cell_indeces = field.create_strip_trial()
    factors = []
    sum = 0
    for j,strip in enumerate(cell_indeces):
        factors.append([i+sum for i, og in enumerate(strip.original_index)])
        sum = sum + len(strip.original_index)
    FA.factors = factors
    FA.get_factor_topology_elements()

    ga = NSGA2
    for population in population_sizes:
        for ga_run in ga_runs:
            start = time.time()
            filename = '/home/alinck/FEA/results/FEAMOO/CCEAMOO_' + field_names[i] + '_trial_3_objectives_strip_topo_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
            feamoo = FEAMOO(Prescription, fea_runs, ga_run, population, FA, ga, dimensions=len(field.cell_list), combinatorial_options=field.nitrogen_list, field=field)
            feamoo.run()
            end = time.time()
            file = open(filename, "wb")
            pickle.dump(feamoo, file)
            elapsed = end-start
            print("FEA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))
