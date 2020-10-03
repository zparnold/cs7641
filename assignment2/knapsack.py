import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    name_of_exp = "Knapsack"
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    max_weight_pct = 0.6
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness, maximize=True, max_val=5)

    # Define initial state
    init_state = np.zeros(5)
    x_s = []
    y_s = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    w_s = []
    max_val = 18.0
    found_flag = False
    for restarts in np.arange(0, 5):
        if found_flag:
            break
        for max_iter_atts in np.arange(10, 1000, 10):
            if found_flag:
                break
            # Solve problem using simulated annealing
            best_state, best_fitness, learning_curve, timing_curve = mlrose.random_hill_climb(problem, max_attempts=int(
                max_iter_atts), max_iters=int(max_iter_atts),
                                                                                              restarts=int(restarts),
                                                                                              init_state=init_state,
                                                                                              curve=True,
                                                                                              random_state=1)
            if best_fitness >= max_val:
                x_s.append(np.arange(0, len(learning_curve)))
                y_s.append(learning_curve)
                w_s.append(timing_curve)
                print(best_state)
                print(best_fitness)
                print(max_iter_atts)
                print(restarts)
                found_flag = True

    found_flag = False
    for sched in [mlrose.ExpDecay(), mlrose.GeomDecay(), mlrose.ArithDecay()]:
        if found_flag:
            break
        for max_iter_atts in np.arange(10, 1000, 10):
            if found_flag:
                break
            best_state, best_fitness, learning_curve, timing_curve = mlrose.simulated_annealing(problem,
                                                                                                max_attempts=int(max_iter_atts),
                                                                                                max_iters=int(max_iter_atts),
                                                                                                schedule=sched,
                                                                                                init_state=init_state,
                                                                                                curve=True,
                                                                                                random_state=1)
            if best_fitness >= max_val:
                x_s.append(np.arange(0, len(learning_curve)))
                y_s.append(learning_curve)
                w_s.append(timing_curve)
                print(best_state)
                print(best_fitness)
                print(max_iter_atts)
                print(sched)
                found_flag = True

    found_flag = False
    for prob in np.arange(0.1, 1.1, 0.1):
        if found_flag:
            break
        for pop_size in np.arange(100, 5000, 100):
            if found_flag:
                break
            for max_iter_atts in np.arange(10, 1000, 10):
                if found_flag:
                    break
                best_state, best_fitness, learning_curve, timing_curve = mlrose.genetic_alg(problem, pop_size=int(pop_size),
                                                                                            mutation_prob=prob,
                                                                                            max_attempts=int(max_iter_atts),
                                                                                            max_iters=int(max_iter_atts),
                                                                                            curve=True,
                                                                                            random_state=1)
                if best_fitness >= max_val:
                    x_s.append(np.arange(0, len(learning_curve)))
                    y_s.append(learning_curve)
                    w_s.append(timing_curve)
                    print(best_state)
                    print(best_fitness)
                    print(max_iter_atts)
                    print(prob)
                    print(pop_size)
                    found_flag = True

    found_flag = False
    for prob in np.arange(0.1, 1.1, 0.1):
        if found_flag:
            break
        for pop_size in np.arange(100, 5000, 100):
            if found_flag:
                break
            for max_iter_atts in np.arange(10, 1000, 10):
                if found_flag:
                    break
                best_state, best_fitness, learning_curve, timing_curve = mlrose.mimic(problem, pop_size=int(pop_size),
                                                                                      keep_pct=prob,
                                                                                      max_attempts=int(max_iter_atts),
                                                                                      max_iters=int(max_iter_atts), curve=True,
                                                                                      random_state=1,
                                                                                      fast_mimic=True)
                if best_fitness >= max_val:
                    x_s.append(np.arange(0, len(learning_curve)))
                    y_s.append(learning_curve)
                    w_s.append(timing_curve)
                    print(best_state)
                    print(best_fitness)
                    print(max_iter_atts)
                    print(prob)
                    print(pop_size)
                    found_flag = True


    for x, y, z in zip(x_s, y_s, z_s):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.title('Randomized Optimization Iterations vs Fitness Function Value for {}'.format(name_of_exp))
    plt.xlabel('Function iteration count')
    plt.ylabel('Fitness function value')
    plt.show()
    plt.clf()
    for x, w, z in zip(x_s, w_s, z_s):
        plt.plot(x, w, label=z)
    plt.legend()
    plt.title('Randomized Optimization Time vs Fitness Function Value for {}'.format(name_of_exp))
    plt.xlabel('Function iteration count')
    plt.ylabel('Time in Seconds')
    plt.show()


if __name__ == '__main__':
    main()
