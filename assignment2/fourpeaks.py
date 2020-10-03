import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    i_s = np.random.randint(2, size=50)
    fitness = mlrose.SixPeaks()
    problem = mlrose.DiscreteOpt(length=len(i_s), fitness_fn=fitness, maximize=True, max_val=2)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    # Define initial state
    init_state = i_s
    x_s = []
    y_s = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    w_s = []
    print('begin hill climb')
    # Solve problem using simulated annealing
    best_state, best_fitness, learning_curve, timing_curve = mlrose.random_hill_climb(problem, max_attempts=10000,curve=True,
                                                                                      random_state=1)
    print(best_state)
    print(best_fitness)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    w_s.append(timing_curve)
    print('begin SA')
    best_state, best_fitness, learning_curve, timing_curve = mlrose.simulated_annealing(problem, max_attempts=250,
                                                                                        max_iters=250,
                                                                                        schedule=schedule,
                                                                                        init_state=init_state,
                                                                                        curve=True, random_state=1)
    print(best_state)
    print(best_fitness)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    w_s.append(timing_curve)
    print('begin GA')
    best_state, best_fitness, learning_curve, timing_curve = mlrose.genetic_alg(problem, pop_size=200,
                                                                                mutation_prob=0.1, max_attempts=250,
                                                                                max_iters=250, curve=True,
                                                                                random_state=1)
    print(best_state)
    print(best_fitness)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    w_s.append(timing_curve)
    print('begin MIMIC')
    best_state, best_fitness, learning_curve, timing_curve = mlrose.mimic(problem, pop_size=250, keep_pct=0.2,
                                                                          max_attempts=250, max_iters=250, curve=True,
                                                                          random_state=1, fast_mimic=True)
    print(best_state)
    print(best_fitness)
    x_s.append(np.arange(0, len(learning_curve)))
    y_s.append(learning_curve)
    w_s.append(timing_curve)
    for x, y, z in zip(x_s, y_s, z_s):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.title('Randomized Optimization Iterations vs Fitness Function Value')
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()
    plt.clf()
    for x, w, z in zip(x_s, w_s, z_s):
        plt.plot(x, w, label=z)
    plt.legend()
    plt.title('Randomized Optimization Time vs Fitness Function Value')
    plt.xlabel('Function iteration count')
    plt.ylabel('Time in Second')
    plt.show()


if __name__ == '__main__':
    main()
