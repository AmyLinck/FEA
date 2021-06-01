# <codecell>
import csv
from copy import copy
from core import Particle

import pso
from fea_common import *


# <markdowncell>

# 1. to what extent can I use the existing PSO code?
# I think that the existing DFEA code can be more easily modified because the main difference
# between the two is how G calculated not in how it's used. For example,
#
# 1. Suppose we have a Swarm, S, with a global solution, G.
# 2. G is a solution for an n dimensional problem but S only optimizes over m dimensions of that problem, m \subset n.
# 3. G fills in the values n - m that S is not optimizing when it needs to
# 4. In this sense, you can think of some function h() that wraps S and G so that f(x_n) works.
# 5. This is going to be the same for both FEA and DFEA, the only different is in how G is calculated.
# 6. Beyond that, G can be used in other ways, for example, we can replace the worst individual in the population
#    with G.

# Basically we need a version of PSO that "wraps" this use of G.
# Strategies:
#
# The PSO runs for N interations 10, 20, or some condition is met.
# Compete and Share happens and we have a G.
# What do we do with G?
# Independent of the pseudocode, we:
#   1. construct h() that closes over G so that x_m can be evaluated using it.
#   2. replace worst individual in S.
#   3. add G to the swarm for record keeping.
#
# What happens in DFEA? It's largely the same thing except that sharing is a bit more complicated,
# may not be complete and each swarm has its own G.
#
# why is the replacement of worst part of the sharing() in FEA. It would seem that would be
# a open implementation detail.
#
def compete(n, swarms, factors, optimizers, f, solution):
    solution = copy(solution)
    variables = list(range(n))
    best_fitness = f(np.array(solution))
    for i in variables:
        best_value = solution[i]
        for swarm_idx in optimizers[i]:
            swarm = swarms[swarm_idx]
            decoder = swarm["d"]
            candidate_x = swarm["gbest"].position[decoder(i)]
            #            print("",swarm_idx,candidate_x)
            solution[i] = candidate_x
            candidate_fitness = f(np.array(solution))
            if candidate_fitness < best_fitness:
                best_fitness = candidate_fitness
                best_value = candidate_x
            # end if
        # end for
        solution[i] = best_value
    #        print("end  ", i, map( lambda v: "%.4f" % v, solution))
    # end for
    # print("best fitness after competition: ", best_fitness)
    return solution

#TODO: RANDOMLY SMASH TOGETHER
def random_compete():
    pass

#TODO: AVERAGE OVER OVERLAPPING VALUES to get var

# Helps with diversity?
def avg_compete():
    pass


# end def

# creates a swarm of length factors with p particles over the domain with
# f as the fitness function. f is defined over n > len( factors) dimensions.
def initialize_fea_swarm(p, n, factors, domain, h):
    # type: (object, object, object, object, object) -> object
    swarm = pso.initialize_swarm(p, len(factors), domain, h)
    swarm["factors"] = factors
    swarm["h"] = h
    # we need to translate from the global home to the local home
    swarm["d"] = make_variable_decoder(factors)
    return swarm


# def

def extract_factors_from_solution(factors, solution):
    return [solution[f] for f in factors]


# end def

def reevaluate_pbests(f, pbests):
    return [Particle(pb.position, pb.velocity, f(np.array(pb.position))) for pb in pbests]


# end def

def find_gbest(pbests):
    gbest = pbests[0]
    for pb in pbests[1:]:
        if pb.fitness < gbest.fitness:
            gbest = pb
    # for
    return gbest


# def

# shares the solution and the implications of the solution with the swarm.
def apply_solution(swarm, solution, f):
    swarm["solution"] = solution
    # make a factor specific fitness function.
    h = make_factored_fitness_fn(swarm["factors"], solution, f)
    swarm["h"] = h

    #    swarm[ "h"] = f

    # so this is weird. All the current pbests were evaluated in the context of the
    # the previous solution so their fitness may or may not be the same under the
    # new solution.
    swarm["pbests"] = reevaluate_pbests(swarm["h"], swarm["pbests"])
    # And this goes for the gbest as well.
    swarm["gbest"] = find_gbest(swarm["pbests"])
    # Replace the worst particle with the solution.
    solution_factors = extract_factors_from_solution(swarm["factors"], solution)
    solution_as_particle = Particle(solution_factors, solution_factors, swarm["h"](solution_factors))
    swarm["particles"][-1] = solution_as_particle  # need to extract out based on factors.
    swarm["pbests"][-1] = solution_as_particle  # need to extract out based on factors.

    return swarm


# end def

def share(swarm, solution, f):
    return apply_solution(swarm, solution, f)


# end def

# update_swarm takes the swarm, f and domain. This is an adapter so that
# we can extract the stuff we've stored in the swarm.
def update_fea_swarm(swarm):
    return pso.update_swarm(swarm, swarm["h"])


# end def

# This is what Shane does...most of the time. Just start out with a random G.
def initialize_solution(n, domain, f):
    particle = pso.initialize_particle(n, domain, f)
    # print("initializtion fitness", particle.fitness)
    return particle.position


# end def

def print_swarms(swarms):
    for i, swarm in enumerate(swarms):
        print(i, swarm)
# end def


# Runs optimization on a single swarm
def optimize_swarm(params):
    swarm, pso_stop = params
    t = 0
    while not pso_stop(t, swarm):
        t = t + 1
        # print("TIME: " + str(t))
        swarm = update_fea_swarm(swarm)

    # thread_lock.acquire()
    # thread_lock.release()

    return swarm


"""
f = function
n = the dimension
domain
factors
optimizors
p = number of particles
fea_times = number of iterations of fea
pso_stop = lambda pso termination function, currently just runs PSO x times
"""
def fea_pso(f, n, domain, all_factors, optimizers, p, fea_times, pso_stop):
    #t_init_start = time.time()
    solution = initialize_solution(n, domain, f)
    solutions = [Particle(position=solution, velocity=[], fitness=f(np.array(solution)))]
    swarms = [initialize_fea_swarm(p, n, factors, domain, make_factored_fitness_fn(factors, solution, f)) for factors in
              all_factors]

    # with just f, this should still work well.
    #   swarms = [initialize_fea_swarm( p, n, factors, domain, f) for factors in all_factors]

    for i in range(fea_times):
        print('fea loop: ', i)
        #t_optimize_start = time.time()
        new_swarms = [None for _ in range(len(swarms))]  # init blank list so no out of bounds errors

        # lock = threading.Lock()  # to make access to new_swarms safe (maybe better way to do this)

        # Optimize each swarm on new thread
        # init the threads to run optimize_swarm(swarm, pso_stop, indx, new_swarms, lock)
        optimize_args = [[swarm, pso_stop] for indx, swarm in enumerate(swarms)]
        # threads = [threading.Thread(target=optimize_swarm, args=(swarm, pso_stop, indx, new_swarms)) for indx, swarm in enumerate(swarms)]

        # pool = NoDaemonProcessPool(len(optimize_args))
        #pool = mp.ThreadingPool(int(mp.cpu_count()/2))
        new_swarms = [optimize_swarm(args) for args in optimize_args]
        #pool.close()
        #pool.join()
        #pool.restart()

        # end for
        swarms = new_swarms
        solution = compete(n, swarms, all_factors, optimizers, f, solution)
        print(solution)

        file = open('results/FEA_PSO/temp/' + 'ODGFunction' + '_dim' + str(
            n) + "m4_diff_grouping_small_epsilon" + ".csv", 'a')
        csv_writer = csv.writer(file)
        a = [i] + solution
        csv_writer.writerow(a)
        file.close()

        swarms = [share(swarm, solution, f) for swarm in swarms]
        solutions.append(Particle(position=solution, velocity=[], fitness=f(np.array(solution))))
    # end for
    # pso.random.reset()
    return solutions


# end def

if __name__ == "__main__":
    from evaluation import create_bootstrap_function, analyze_bootstrap_sample
    from topology import generate_linear_topology
    import json

    p = 10 # population
    d = 32 # dimensions
    k = 2 # factor size
    i = 5 # pso stop
    j = 20 # fea times

    factors, arbiters, optimizers, neighbors = generate_linear_topology(d, k)

    names = list(benchmarks.keys())
    names.sort()

    for name in names[:1]:
        with open("results/" + name + "-fea.json", "w") as outfile:
            print("working on", name)
            summary = {"name": name}
            fitnesses = []
            for trial in range(0, 50):
                benchmark = benchmarks[name]
                f = benchmark["function"]
                domain = benchmark["interval"]

                #                random.seed(seeds[trial])
                result = fea_pso(f, d, domain, factors, optimizers, p, j, lambda t, f: t == i)
                fitnesses.append(result[-1].fitness)
            bootstrap = create_bootstrap_function(250)
            replications = bootstrap(fitnesses)
            statistics = analyze_bootstrap_sample(replications)
            summary["statistics"] = statistics
            summary["fitnesses"] = fitnesses
            json.dump(summary, outfile)
#