from refactoring.optimizationproblems.moo_functions import MOOFunctions
from refactoring.FEA.factorarchitecture import MooFactorArchitecture

dim = 20
n_obj = 3
# param h tells teh benchmark functions how many reference points to create for each objective
h = 2
dtlz = MOOFunctions(dim, n_obj, h)
# param decomp_approach = string of the function name to call for decomposition, e.g.: diff_grouping, MEET, etc.
MFA = MooFactorArchitecture(dim, dtlz, decomp_approach='overlapping_diff_grouping')
fa = MFA.create_objective_factors()  # returns regular FactorArchitecture class instance
