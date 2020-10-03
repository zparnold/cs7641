import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
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
    fitness = mlrose.CustomFitness(queens_max)
    problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    x_s = []
    y_s = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    max_val = 28.0
    found_flag = False
    for restarts in np.arange(0,5):
        if found_flag:
            break
        for max_iter_atts in np.arange(10, 1000, 10):
            if found_flag:
                break
            # Solve problem using simulated annealing
            best_state, best_fitness, learning_curve, timing_curve = mlrose.random_hill_climb(problem, max_attempts=int(max_iter_atts), max_iters=int(max_iter_atts),
                                                                                restarts=int(restarts), init_state=init_state, curve=True,
                                                                                random_state=1)
            if best_fitness == max_val:
                x_s.append(np.arange(0, len(learning_curve)))
                y_s.append(learning_curve)
                print(best_state)
                print(best_fitness)
                print(max_iter_atts)
                print(restarts)
                found_flag = True


    best_state, best_fitness, learning_curve, timing_curve = mlrose.simulated_annealing(problem, max_attempts=100, max_iters=1000,
                                                                          schedule=schedule, init_state=init_state,
                                                                          curve=True, random_state=1)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    print(best_state)
    print(best_fitness)
    best_state, best_fitness, learning_curve, timing_curve = mlrose.genetic_alg(problem, pop_size=2000, mutation_prob=0.1,
                                                                  max_attempts=100, max_iters=1000, curve=True,
                                                                  random_state=1)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    print(best_state)
    print(best_fitness)
    best_state, best_fitness, learning_curve, timing_curve = mlrose.mimic(problem, pop_size=2000, keep_pct=0.2, max_attempts=100,
                                                            max_iters=1000, curve=True, random_state=1,
                                                            fast_mimic=True)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    print(best_state)
    print(best_fitness)
    for x, y, z in zip(x_s, y_s, z_s):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.title('Randomized Optimization Iterations vs Fitness Function Value')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()


if __name__ == '__main__':
    main()
