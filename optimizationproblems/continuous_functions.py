#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#
# Modified by Elliott Pryor 08 March 2021 & Amy Peerlinck Apr 16 2021
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from numpy.random import seed, permutation
from numpy import dot, ones
from optimizationproblems.benchmarks import *
import math
import random

# CEC 2010 Benchmark Problems
class BenchmarkFunction(object):

    # (lbound, ubound)
    @staticmethod
    def default_bounds(function_number):
        if function_number % 3 == 0: # 3,6,9,...
            return (-32,32)
        elif function_number % 3 == 1: # 1,4,7,10,13,16
            return (-100,100)
        elif function_number % 3 == 2: # 2,5,8,11,14,17
            return (-5,5)

    # "D" = dim
    @staticmethod
    def default_dimension(function_number):
        return 1000


    # "m" = m_shift
    @staticmethod
    def default_group_size(function_number,N):
        if N == 1000:
            return 50
        else:
            return int(np.clip(math.ceil(N/20.0),1,N))
        
    function_names = {
        1 : "Shifted Elliptic Function",
        2 : "Shifted Rastrigin’s Function",
        3 : "Shifted Ackley’s Function",
        4 : "Single-group Shifted and m-rotated Elliptic Function",
        5 : "Single-group Shifted and m-rotated Rastrigin’s Function",
        6 : "Single-group Shifted and m-rotated Ackley’s Function",
        7 : "Single-group Shifted m-dimensional Schwefel’s Problem 1.2",
        8 : "Single-group Shifted m-dimensional Rosenbrock’s Function",
        9 : "D/2m-group Shifted and m-rotated Elliptic Function",
        10 : "D/2m-group Shifted and m-rotated Rastrigin’s Function",
        11 : "D/2m-group Shifted and m-rotated Ackley’s Function",
        12 : "D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2",
        13 : "D/2m-group Shifted m-dimensional Rosenbrock’s Function",
        14 : "D/2m-group Shifted and m-rotated Elliptic Function",
        15 : "D/2m-group Shifted and m-rotated Rastrigin’s Function",
        16 : "D/2m-group Shifted and m-rotated Elliptic Function",
        17 : "D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2",
        18 : "D/2m-group Shifted m-dimensional Rosenbrock’s Function",
        19 : "Shifted Schwefel’s Problem 1.2",
        20 : "Shifted Rosenbrock’s Function"
    }

    def function_name(self,function_number):
        if function_number in BenchmarkFunction.function_names:
            return BenchmarkFunction.function_names[function_number]
        else:
            return self.function_to_call
        
    def __init__(self, function_number=0, shift_data=None, shift_data_file="", matrix_data=None, matrix_data_file="", **kwargs):
        
        self.function_number = function_number
        self.function_evaluations = 0
        # self.counter_lock = threading.Lock()
    
        self.function_to_call = 'F'+str(function_number)
        self.derivative_to_call = 'D'+str(function_number)
        self.function_call = getattr(self, self.function_to_call)
        # self.derivative_call = getattr(self, self.derivative_to_call)

        # Set default benchmark function parameters
        self.dimensions = self.default_dimension(self.function_number)
        self.lbound, self.ubound = self.default_bounds(self.function_number)
        self.m_shift = self.default_group_size(self.function_number, self.dimensions)
        self.name = self.function_name(self.function_number)

        # Override default benchmark values
        if "dim" in kwargs:
            self.dimensions = kwargs["dim"]

        if "lbound" in kwargs or "rbound" in kwargs:
            if "lbound" in kwargs and "rbound" in kwargs:
                self.lbound = kwargs["lbound"]
                self.rbound = kwargs["rbound"]
            else:
                raise ValueError("lbound and rbound must be specified")

        if "m" in kwargs:
            self.m_shift = kwargs["m"]


        # Instance of random to generate random shift and random permutation
        self.seed = None
        if "seed" in kwargs:
            self.seed = kwargs["seed"]

        if "random" in kwargs:
            self.random = kwargs["random"]
        elif self.seed is None:
            self.random = random.Random()
        else:
            self.random = random.Random(self.seed)

        if "nprandom" in kwargs:
            self.nprandom = kwargs["nprandom"]
        elif self.seed is None:
            self.nprandom = np.random.RandomState()
        else:
            self.nprandom = np.random.RandomState(self.seed)




        # Set shift and permutation data
        self.shift_data = shift_data
        self.matrix_data = matrix_data # Rotational matrix
        self.permu_data = None

        # # Deprecated by Nathan, replaced with after
        # if shift_data_file != "" and matrix_data_file == "":
        # 	if 4 > function_number or (18 < function_number < 21):
        # 		self.shift_data = load_shift_data__(shift_data_file)
        # 	else:
        # 		self.shift_data = load_matrix_data__(shift_data_file)
        # elif matrix_data_file != "":
        # 	self.matrix_data = load_matrix_data__(matrix_data_file)
        # 	self.shift_data = load_matrix_data__(shift_data_file)


        # Below now does what shift_permutation did,
        # However, I did not verify correctness with files yet.
        if shift_data is None:
            if shift_data_file == "":
                self.shift_data = self.nprandom.uniform(-self.lbound, self.ubound, size=self.dimensions)
            else:
                from opfunu.cec.cec2010.utils import load_shift_data__
                self.shift_data = load_shift_data__(shift_data_file)
        else:
            self.shift_data = self.shift_data[:1, :].reshape(-1)[:self.dimensions]
            if self.dimensions == 1000: # permu_data comes from when first row when the data file is provided
                self.permu_data = (self.shift_data[1:, :].reshape(-1) - ones(self.dimensions)).astype(int)

        if matrix_data is None:
            if matrix_data_file == "":
                import scipy
                self.matrix_data = scipy.stats.special_ortho_group(self.dimensions, seed=self.nprandom)
            else:
                from opfunu.cec.cec2010.utils import load_matrix_data__
                self.matrix_data = load_matrix_data__(matrix_data_file)
            
        if self.permu_data is None:
            self.permu_data = self.nprandom.permutation(self.dimensions)

    def reset_counter(self):
        # with self.counter_lock:
        self.function_evaluations = 0

    def run(self, solution):
        # with self.counter_lock:
        self.function_evaluations += 1
        if self.dimensions == 0:
            self.dimensions = len(solution)
            # check_problem_size(self.dimensions)
        return self.function_call(solution=solution)

    def grad_estimate(self,solution,factor=None):
        from gradient_estimation import GradientEstimate # Inefficient if placed here

        if factor is None:
            self.factor = list(range(0,self.dimensions))
        else:
            self.factor = factor

        self.factor = np.array(self.factor)

        # warnings.warn("Gradient num of sample points untuned")
        numpoints = 2*min(len(self.factor),self.dimensions) # 2 points in every dimension?
        estimator = GradientEstimate(0.002,numpoints,self.factor)
        grad = estimator.gradient(self.run,solution,self.factor)
        return grad


    
    def gradient(self,solution):
        if self.dimensions == 0:
            self.dimensions = len(solution)
        return self.derivative_call(solution=solution)
        # return getattr(self, self.derivative_to_call)(solution=solution)

    # Precalculated now in __init__ instead of every function call
    def shift_permutation(self):
        return self.shift_data, self.permu_data

    # Test
    def F0(self, solution=None):
        s = np.sum(solution**2)
        return s

    # def F1(self, solution=None, name="Shifted Elliptic Function"):
    def F1(self, solution=None):
        z = solution - self.shift_data[:self.dimensions]
        return elliptic__(z,dim=self.dimensions)

    # def F2(self, solution=None, name="Shifted Rastrigin’s Function"):
    def F2(self, solution=None):
        z = solution - self.shift_data[:self.dimensions]
        return rastrigin__(z)

    # def F3(self, solution=None, name="Shifted Ackley’s Function"):
    def F3(self, solution=None):
        z = solution - self.shift_data[:self.dimensions]
        return ackley__(z)
    
    #def F4(self, solution=None, name="Single-group Shifted and m-rotated Elliptic Function", m_group=50):
    def F4(self, solution=None):
        m_group = self.m_shift
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_elliptic = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_elliptic = z[idx2]
        return elliptic__(z_rot_elliptic) * 10**6 + elliptic__(z_elliptic)
    
    # f(g(x))=f'(g(x))g'(x) = df_dg * dg_dxmatrix_data
    
    # def D4(self, solution=None, name="Single-group Shifted and m-rotated Elliptic Function", m_group=50):
        # self.name = name
        # shift_data, permu_data = self.shift_data, self.perm_data
        # idx1 = permu_data[:m_group]
        # idx2 = permu_data[m_group:]

        # z = solution - shift_data

        # M = self.matrix_data[:m_group, :m_group]
        # z_rot_elliptic = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        # z_elliptic = z[idx2]

        # ## UNPERMUTATE and RECOMBINE
        # df_dg = d_elliptic__(z_rot_elliptic)*np.transpose() * 10**6 + d_elliptic__(z_elliptic)
        # dg_dx = 

        # return dg_dx
        
    


    #def F5(self, solution=None, name="Single-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
    def F5(self, solution=None):
        m_group = self.m_shift
        shift_data, permu_data = self.shift_data, self.permu_data
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_rastrigin = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_rastrigin = z[idx2]
        return rastrigin__(z_rot_rastrigin) * 10 ** 6 + rastrigin__(z_rastrigin)

    # def D5(self,solution=None):
    # 	grad_tanh = grad(self.F5)
    # 	return grad_tanh(solution)
    
    #def F6(self, solution=None, name="Single-group Shifted and m-rotated Ackley’s Function", m_group=50):
    def F6(self, solution=None):
        m_group = self.m_shift
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_ackley = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_ackley = z[idx2]
        return ackley__(z_rot_ackley) * 10 ** 6 + ackley__(z_ackley)
       
    # def F7(self, solution=None, name="Single-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=50):
    def F7(self, solution=None):
        m_group = self.m_shift
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_schwefel = z[idx1]
        z_shpere = z[idx2]
        return schwefel__(z_schwefel) * 10 ** 6 + sphere__(z_shpere)
      
    # def F8(self, solution=None, name=" Single-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
    def F8(self, solution=None):
        m_group = self.m_shift
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rosenbrock = z[idx1]
        z_sphere = z[idx2]
        return rosenbrock__(z_rosenbrock) * 10 ** 6 + sphere__(z_sphere)
      
    # def F9(self, solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", m_group=50):
    def F9(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F9", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i*m_group:(i+1)*m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += elliptic__(z1)
        idx2 = permu_data[int(self.dimensions/2):self.dimensions]
        z2 = z[idx2]
        result += elliptic__(z2)
        return result
       
    # def F10(self, solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
    def F10(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F10", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += rastrigin__(z1)
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        z2 = z[idx2]
        result += rastrigin__(z2)
        return result
     
    # def F11(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", m_group=50):
    def F11(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F11", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += ackley__(z1)
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        z2 = z[idx2]
        result += ackley__(z2)
        return result
      
    # def F12(self, solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=50):
    def F12(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F12", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += schwefel__(z[idx1])
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        result += sphere__(z[idx2])
        return result
      
    # def F13(self, solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
    def F13(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F13", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rosenbrock__(z[idx1])
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        result += sphere__(z[idx2])
        return result
     
    # def F14(self, solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", m_group=50):
    def F14(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / m_group)
        # check_m_group("F14", self.dimensions, m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += elliptic__(dot(z[idx1], self.matrix_data))
        return result
      
    # def F15(self, solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
    def F15(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / m_group)
        # check_m_group("F15", self.dimensions, m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rastrigin__(dot(z[idx1], self.matrix_data))
        return result
  
    # def F16(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", m_group=50):
    def F16(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / m_group)
        # check_m_group("F16", self.dimensions, m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += ackley__(dot(z[idx1], self.matrix_data))
        return result
  
    # def F17(self, solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=4):
    def F17(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / m_group)
        # check_m_group("F17", self.dimensions, m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += schwefel__(z[idx1])
        return result

    # def F18(self, solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
    def F18(self, solution=None):
        m_group = self.m_shift
        epoch = int(self.dimensions / m_group)
        # check_m_group("F18", self.dimensions, m_group)
        shift_data, permu_data = self.shift_data, self.perm_data
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rosenbrock__(z[idx1])
        return result
   
    # def F19(self, solution=None, name="Shifted Schwefel’s Problem 1.2"):
    def F19(self, solution=None):
        m_group = self.m_shift
        shift_data = self.shift_data[:self.dimensions]
        z = solution - shift_data
        return schwefel__(z)
    
    # def F20(self, solution=None, name="Shifted Rosenbrock’s Function"):
    def F20(self, solution=None):
        m_group = self.m_shift
        shift_data = self.shift_data[:self.dimensions]
        z = solution - shift_data
        return rosenbrock__(z)



if __name__ == '__main__':
    f = BenchmarkFunction(0)

    x = np.zeros((1000,),dtype="float64")
    f.run(x)
    # f.grad_estimate(x)