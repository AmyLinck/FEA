import numpy as np
import random
from operator import attrgetter
from copy import deepcopy, copy


class Particle(object):
    def __init__(self, function, dim, position=None, factor=None, global_solution=None, lbest_pos=None, factorid=None, alg=None):
        self.alg = alg
        self.f = function
        self.lbest_fitness = float('inf')
        self.dim = dim
        self.factor = factor
        if self.factor is None:
            self.factor = [i for i in range(dim)]
        self.factorid = factorid
        if position is None:
            self.position = np.random.uniform(self.f.lbound, self.f.ubound, size=dim)
            self.lbest_position = np.array([x for x in self.position])
        elif position is not None:
            self.position = position
            self.lbest_position = lbest_pos
            self.lbest_fitness = self.calculate_fitness(global_solution, lbest_pos)
        self.velocity = np.zeros(dim)
        self.set_fitness(self.calculate_fitness(global_solution))

    def __le__(self, other):
        if self.fitness is float:
            return self.fitness <= other.fitness

    def __lt__(self, other):
        if self.fitness is float:
            return self.fitness < other.fitness

    def __gt__(self, other):
        if self.fitness is float:
            return self.fitness > other.fitness

    def __eq__(self, other):
        return (self.position == other.position).all()

    def __str__(self):
        return ' '.join(
            ['Particle with current fitness:', str(self.fitness), 'and best fitness:', str(self.lbest_fitness)])

    def set_fitness(self, fit):
        self.fitness = fit
        if fit < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fit)
            self.lbest_position = np.array([x for x in self.position])

    def set_position(self, position):
        self.position = np.array(position)

    def get_position(self):
        return np.clip(self.position, self.f.lbound, self.f.ubound)

    def update_individual_after_compete(self, global_solution=None):
        return self.update_position(global_solution)
    
    def calc_position(self, glob_solution=None, position=None):
        if position is None:
            position = self.position
        if glob_solution is None:
            glob_solution = self.position

        solution = np.array([x for x in glob_solution])
        for i, x in zip(self.factor, position):
            solution[i] = x
        return solution

    def calculate_fitness(self, glob_solution=None,position=None):
        solution = self.calc_position(glob_solution,position)
        solution = np.clip(solution, self.f.lbound, self.f.ubound)
        fitness = self.f.run(np.array(solution))
        return fitness

    def update_particle(self, omega, phi, global_best_position, v_max, global_solution=None):
        self.update_velocity(omega, phi, global_best_position, v_max)
        self.update_position(global_solution)

    def update_velocity(self, omega, phi, global_best_position, v_max):
        velocity = [x for x in self.velocity]
        n = self.dim

        inertia = np.multiply(omega, velocity)
        phi_1 = np.array([random.random() * phi for i in range(n)])  # exploration
        personal_exploitation = self.lbest_position - self.position  # exploitation
        personal = phi_1 * personal_exploitation

        phi_2 = np.array([random.random() * phi for i in range(n)])  # exploration
        social_exploitation = global_best_position - self.position  # exploitation
        social = phi_2 * social_exploitation
        
        new_velocity = inertia + personal + social
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])

    def update_position(self, global_solution=None):
        position = self.velocity + self.position
        # Clamping can cause some unit velocities to be 0, especially in high dimensional spaces
        self.position = np.clip(position, self.f.lbound, self.f.ubound)
        # self.position = position
        self.set_fitness(self.calculate_fitness(global_solution))

    def clamp_value(self, to_clamp, lo, hi):
        if lo < to_clamp < hi:
            return to_clamp
        if to_clamp < lo:
            return lo
        return hi


class PSO(object):
    def __init__(self, generations, population_size, function, dim, factor=None, global_solution=None, factorid=None, omega=0.729, phi=1.49618):
        self.pop_size = population_size
        self.pop = [Particle(function, dim, factor=factor, global_solution=global_solution,factorid=factorid, alg=self) for x in range(population_size)]
        pos = [p.position for p in self.pop]

        # with open('pso2.o', 'a') as file:
        #     file.write(str(pos))
        #     file.write('\n')

        self.omega = omega
        self.phi = phi
        self.f = function
        self.dim = dim
        pbest_particle = Particle(function, dim, factor=factor, global_solution=global_solution, alg=self)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((function.ubound - function.lbound))
        self.generations = generations
        self.current_loop = 0
        if factor is None:
            factor = [i for i in range(dim)]
        self.factor = np.array(factor)
        self.factorid = factorid
        self.global_solution = global_solution

    def find_current_best(self):
        sorted_ = sorted(np.array(self.pop), key=attrgetter('fitness'))
        return Particle(self.f, self.dim, position=sorted_[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=sorted_[0].lbest_position,factorid=sorted_[0].factorid, alg=self)

    def find_local_best(self):
        pass

    def update_swarm(self):
        if self.global_solution is not None:
            global_solution = [x for x in self.global_solution]
        else:
            global_solution = None
        omega, phi, v_max = self.omega, self.phi, self.v_max
        global_best_position = [x for x in self.gbest.position]
        for p in self.pop:
            p.update_particle(omega, phi, global_best_position, v_max, global_solution)
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def replace_worst_solution(self, global_solution):
        # find worst particle
        self.global_solution = np.array([x for x in global_solution])
        self.pop.sort(key=attrgetter('fitness'))
        # print('replacing')
        # print(self.pop[-1], self.pop[0])
        partial_solution = [x for i, x in enumerate(global_solution) if i in self.factor] # if i in self.factor
        self.pop[-1].set_position(partial_solution)
        self.pop[-1].set_fitness(self.f.run(self.global_solution))
        curr_best = Particle(self.f, self.dim, position=self.pop[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=self.pop[0].lbest_position, alg=self)
        random.shuffle(self.pop)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def run(self):
        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            # print(self.gbest)
            idx = np.argmax(self.gbest.position)
            p = self.pop[0].position[idx]
            v = self.pop[0].velocity[idx]
            # print(self.pop[0].fitness)
            # print(i,idx,p,v,sep="\n",end="\n\n")
            # print(i,self.gbest.fitness,np.linalg.norm(self.pop[0].position),np.linalg.norm(self.pop[0].position),sep="\n",end="\n\n")
        return self.gbest.position


## TESTING PSO
if __name__ == '__main__':
    from optimizationproblems.continuous_functions import BenchmarkFunction

    dim=100
    popsize=100
    gens=1000

    f = BenchmarkFunction(function_number=0,dim=dim)
    pso = PSO(generations=gens, population_size=popsize, function=f, dim=dim)
    pso.run()
    # [print(x) for x in pso.pop]
