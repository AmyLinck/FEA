import gc
from refactoring.utilities.util import PopulationMember

from ..MOO.paretofront import ParetoOptimization
from ..FEA.factorarchitecture import FactorArchitecture
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

import numpy as np
import random


class FEAMOO:
    def __init__(self, fea_iterations, alg_iterations, pop_size, fa, base_alg, dimensions,
                 combinatorial_options=[], ref_point = [1e30,1e30,1e30]):
        self.combinatorial_options = combinatorial_options
        self.dim = dimensions
        self.nondom_archive = []
        self.population = []
        self.factor_architecture = fa
        self.base_algorithm = base_alg
        self.fea_runs = fea_iterations
        self.base_alg_iterations = alg_iterations
        self.pop_size = pop_size
        self.current_iteration = 0
        self.global_solutions = []
        self.worst_fitness_ref = ref_point
        self.po = ParetoOptimization()
        # keep track to have a reference point for the HV indicator
        self.subpopulations = self.initialize_moo_subpopulations()
        self.iteration_stats = []

    def initialize_moo_subpopulations(self):
        random_global_variables = random.choices(self.combinatorial_options, k=self.dim)
        objs = self.base_algorithm().calc_fitness(random_global_variables)
        random_global_solution = PopulationMember(random_global_variables, objs)
        self.global_solutions.append(random_global_solution)
        return [self.base_algorithm(ea_runs=self.base_alg_iterations, dimensions=len(factor), combinatorial_values=self.combinatorial_options, population_size=self.pop_size, factor=factor,
                    global_solution=random_global_solution) for factor in self.factor_architecture.factors]

    def update_archive(self):
        nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in self.nondom_archive]))
        nondom_archive = [self.nondom_archive[i] for i in nondom_indeces]
        seen = set()
        self.nondom_archive = [seen.add(s.fitness) or s for s in nondom_archive if s.fitness not in seen]
        print(seen)
        [print(s.fitness) for s in self.nondom_archive]

    def run(self):
        '''
        For each subpopulation:
            Optimize along all objectives
            Apply non-dominated sorting strategy
            Include diversity measure individuals
        Compete (based on non-domination of solution, i.e. overall fitness)
        -> or ignore compete and create set of solutions based on overlapping variables: improves diversity? Check this with diversity measure
        -> spread different non-domination solutions across different subpopulations, i.e., different subpopulations have different global solutions: this should also improve diversity along the PF?
        @return: eval_dict {HV, diversity, ND_size, FEA_run}
        '''
        change_in_nondom_size = []
        old_archive_length = 0
        fea_run = 0
        while len(change_in_nondom_size) < 4 and fea_run != self.fea_runs:
            for alg in self.subpopulations:
                alg.run()
            self.compete()
            self.share_solution()
            self.update_archive()
            if len(self.nondom_archive) == old_archive_length:
                change_in_nondom_size.append(True)
            else:
                change_in_nondom_size = []
            print('archive length: ', len(self.nondom_archive))
            # [print(x.fitness) for x in self.nondom_archive]
            print(self.nondom_archive[-1].fitness)
            old_archive_length = len(self.nondom_archive)
            eval_dict = self.po.evaluate_solution(self.nondom_archive, self.worst_fitness_ref)
            eval_dict['FEA_run'] = fea_run
            eval_dict['ND_size'] = len(self.nondom_archive)
            self.iteration_stats.append(eval_dict)
            print(eval_dict)
            # [print(s.objective_values) for s in self.nondom_archive]
            # [print(i, ': ', s.objective_values) for i,s in enumerate(self.iteration_stats[fea_run+1]['global solutions'])]
            fea_run = fea_run+1

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        new_solutions = []
        for var_idx in range(self.dim):
            # randomly pick one of the global solutions to perform competition for this variable
            chosen_global_solution = random.choice(self.global_solutions)
            vars = [x for x in chosen_global_solution.variables]
            #sol = PopulationMember(vars, self.base_algorithm().calc_fitness(vars))
            if not new_solutions:
                new_solutions.append(vars)
            # for each population with said variable perform competition on this single randomly chosen global solution
            if len(self.factor_architecture.optimizers[var_idx]) > 1:
                for pop_idx in self.factor_architecture.optimizers[var_idx]:
                    curr_pop = self.subpopulations[pop_idx]
                    pop_var_idx = np.where(np.array(curr_pop.factor) == var_idx)
                    # randomly pick one of the nondominated solutions from this population
                    if len(curr_pop.nondom_pop) != 0:
                        sorted = curr_pop.diversity_sort(curr_pop.nondom_pop)
                        random_sol = sorted[0]
                    else:
                        random_sol = random.choice(curr_pop.gbests)
                    var_candidate_value = random_sol.variables[pop_var_idx[0][0]]
                    vars[var_idx] = var_candidate_value
                    if vars not in new_solutions:
                        new_solutions.append(vars)
            elif len(self.factor_architecture.optimizers[var_idx]) == 1:
                curr_pop = self.subpopulations[self.factor_architecture.optimizers[var_idx][0]]
                pop_var_idx = np.where(np.array(curr_pop.factor) == var_idx)
                if len(curr_pop.nondom_pop) != 0:
                    for solution in curr_pop.nondom_pop:
                        var_candidate_value = solution.variables[pop_var_idx[0][0]]
                        vars[var_idx] = var_candidate_value
                        if vars not in new_solutions:
                            new_solutions.append(vars)
                # if sol < chosen_global_solution:
                #     best_value_for_var = var_candidate_value
            # sol.variables[var_idx] = best_value_for_var
            # new_solutions.append(sol)
        new_solutions = [PopulationMember(vars, self.base_algorithm().calc_fitness(vars)) for vars in new_solutions]
        nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in new_solutions]))
        self.global_solutions = [new_solutions[i] for i in nondom_indeces]
        self.nondom_archive.extend(self.global_solutions)

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        if len(self.global_solutions) != 0:
            to_pick = [s for s in self.global_solutions]
        else:
            to_pick = [s for s in self.nondom_archive]
        for i, alg in enumerate(self.subpopulations):
            if len(to_pick) > 1:
                gs = random.choice(to_pick)
                to_pick.remove(gs)
            elif len(to_pick) == 1:
                gs = to_pick[0]
                # repopulate the list to restart if not at the last subpopulation
                if i < len(self.subpopulations) - 1:
                    if len(self.global_solutions) != 0:
                        to_pick = [s for s in self.global_solutions]
                    else:
                        to_pick = [s for s in self.nondom_archive]
            else:
                print('there are no elements in the global solutions list.')
                raise IndexError
            # update fitnesses
            alg.global_solution = gs
            alg.curr_population = [PopulationMember(p.variables, self.base_algorithm().calc_fitness(p.variables, alg.global_solution, alg.factor)) for p in alg.curr_population]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(self.global_solutions)
