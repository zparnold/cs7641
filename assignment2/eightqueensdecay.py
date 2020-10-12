import six
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.modules['sklearn.externals.six'] = six
import mlrose as mlrose
import numpy as np
import math
import matplotlib.pyplot as plt

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    # Initialize counter
    fitness = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):

            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness += 1

    return fitness

def main():
    name_of_exp = "Eight Queens"
    fitness = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

    # Define initial state
    mimic = []
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    for i in [mlrose.ExpDecay(), mlrose.GeomDecay(), mlrose.ArithDecay()]:
        best_state, best_fitness, learning_curve, timing_curve = mlrose.simulated_annealing(problem, init_state=init_state,
                                                                              schedule=i,
                                                                              max_attempts=1000,
                                                                              max_iters=1000, curve=True,
                                                                              random_state=1)
        mimic.append(learning_curve)
        print(i)
        print(best_fitness)
    for x, z in zip(['Exp','Geom','Arith'], mimic):
        plt.plot(z, label=str(x))
    plt.legend()
    plt.title('SA Randomized Optimization DecaySchedule vs Fitness Curve (8-Queens)')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()

if __name__ == '__main__':
    main()
