import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    max_weight_pct = 0.6
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness, maximize=False, max_val=2)

    # Define initial state
    init_state = np.array([1, 0, 2, 1, 0])
    x_s = []
    y_s = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    # Solve problem using simulated annealing
    best_state, best_fitness, learning_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=100,
                                                                        restarts=0, init_state=init_state, curve=True,
                                                                        random_state=1)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    best_state, best_fitness, learning_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=100,
                                                                          schedule=schedule, init_state=init_state,
                                                                          curve=True, random_state=1)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)

    best_state, best_fitness, learning_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=100, max_iters=100, curve=True, random_state=1)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)

    best_state, best_fitness, learning_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=100, max_iters=100, curve=True, random_state=1, fast_mimic=False)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)

    for x, y, z in zip(x_s, y_s, z_s):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.title('Randomized Optimization Iterations vs Fitness Function Value')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()


if __name__ == '__main__':
    main()
