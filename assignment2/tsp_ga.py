import six
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.modules['sklearn.externals.six'] = six
import mlrose as mlrose
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    name_of_exp = "One Max"
    # Create list of city coordinates
    coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
    mimic = []
    # Initialize fitness function object using coords_list
    fitness_coords = mlrose.TravellingSales(coords=coords_list)
    problem = mlrose.TSPOpt(length=8, fitness_fn=fitness_coords,
                            maximize=False)
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    for i in [0.1,0.2,0.3,0.4,0.5]:
        best_state, best_fitness, learning_curve, timing_curve = mlrose.genetic_alg(problem, pop_size=100,
                                                                              mutation_prob=i,
                                                                              max_attempts=100,
                                                                              max_iters=100, curve=True,
                                                                              random_state=1)
        mimic.append(learning_curve)
        print(i)
        print(best_fitness)
    for x, z in zip([0.1,0.2,0.3,0.4,0.5], mimic):
        plt.plot(z, label=str(x))
    plt.legend()
    plt.title('GA Randomized Optimization MutationProb vs Fitness Curve (TSP)')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()

if __name__ == '__main__':
    main()
