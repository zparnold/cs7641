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
    fitness = mlrose.OneMax()
    mimic = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    for i in [100, 200, 300, 400, 500]:
        problem = mlrose.DiscreteOpt(length=15, fitness_fn=fitness, maximize=True, max_val=2)
        print("MIMC")
        max_atts = 10
        best_state, best_fitness, learning_curve, timing_curve = mlrose.mimic(problem, pop_size=i,
                                                                              keep_pct=0.1,
                                                                              max_attempts=100,
                                                                              max_iters=100, curve=True,
                                                                              random_state=1,
                                                                              fast_mimic=True)
        mimic.append(learning_curve)
        print(i)
        print(best_fitness)
        print(max_atts)
    for x, z in zip([100,200,300,400,500], mimic):
        plt.plot(z, label=str(x))
    plt.legend()
    plt.title('MIMIC Randomized Optimization PopSize vs Fitness Curve (OneMax)')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()

if __name__ == '__main__':
    main()
