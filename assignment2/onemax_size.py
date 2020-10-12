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
    rhc = []
    sa = []
    ga = []
    mimic = []
    z_s = ['RHC', 'SA', 'GA', 'MIMIC']
    for i in [5, 10, 15]:
        problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=2)
        # Define initial state
        init_state = np.zeros(i)

        # Solve problem using simulated annealing
        best_fitness = 0
        learning_curve = []
        max_atts = 10
        print("RHC")
        while best_fitness != i:
            best_state, best_fitness, learning_curve, timing_curve = mlrose.random_hill_climb(problem,
                                                                                              max_attempts=max_atts,
                                                                                              max_iters=max_atts,
                                                                                              restarts=1,
                                                                                              init_state=init_state,
                                                                                              curve=True,
                                                                                              random_state=1)
            max_atts += 10

        rhc.append(learning_curve)
        print(i)
        print(best_fitness)
        print(max_atts)
        print("SA")
        best_fitness = 0
        learning_curve = []
        max_atts = 10
        while best_fitness != i:
            best_state, best_fitness, learning_curve, timing_curve = mlrose.simulated_annealing(problem,
                                                                                                max_attempts=max_atts,
                                                                                                max_iters=max_atts,
                                                                                                schedule=mlrose.ExpDecay(),
                                                                                                init_state=init_state,
                                                                                                curve=True,
                                                                                                random_state=1)
            max_atts += 10

        sa.append(learning_curve)
        print(i)
        print(best_fitness)
        print(max_atts)
        print("GA")
        best_fitness = 0
        learning_curve = []
        max_atts = 10
        while best_fitness != i:
            best_state, best_fitness, learning_curve, timing_curve = mlrose.genetic_alg(problem, pop_size=100,
                                                                                        mutation_prob=0.1,
                                                                                        max_attempts=max_atts,
                                                                                        max_iters=max_atts,
                                                                                        curve=True,
                                                                                        random_state=1)
            max_atts += 10
        ga.append(learning_curve)
        print(i)
        print(best_fitness)
        print(max_atts)
        print("MIMC")
        best_fitness = 0
        learning_curve = []
        max_atts = 10
        while best_fitness != i:
            best_state, best_fitness, learning_curve, timing_curve = mlrose.mimic(problem, pop_size=300,
                                                                                  keep_pct=0.1,
                                                                                  max_attempts=max_atts,
                                                                                  max_iters=max_atts, curve=True,
                                                                                  random_state=1,
                                                                                  fast_mimic=True)
            max_atts += 10
        mimic.append(learning_curve)
        print(i)
        print(best_fitness)
        print(max_atts)
    f, axarr = plt.subplots(1, 4)
    f.set_figheight(3)
    f.set_figwidth(12)
    for y in rhc:
        for i in ['5','10','15']:
                axarr[0].plot(y, label='{}'.format(i))
    axarr[0].set_title('RHC vs Input Size')
    for y in sa:
        for i in ['5', '10', '15']:
            axarr[1].plot(y, label='{}'.format(i))
    axarr[1].set_title('SA vs Input Size')
    for y in ga:
        for i in ['5', '10', '15']:
            axarr[2].plot(y, label='{}'.format(i))
    axarr[2].set_title('GA vs Input Size')
    for y in mimic:
        for i in ['5', '10', '15']:
            axarr[3].plot(y, label='{}'.format(i))
    axarr[3].set_title('MIMC vs Input Size')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axarr[0]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.legend(handles=[Line2D([0], [0], color='g', lw=4, label='5'),
                   Line2D([0], [0], color='brown', lw=4, label='10'),
                   Line2D([0], [0], color='y', lw=4, label='15')])
    #plt.title('Input size vs fitness curve One Max')
   # plt.xlabel('Function iteration count')
    #plt.ylabel('Fitness function value')

    plt.show()

if __name__ == '__main__':
    main()
