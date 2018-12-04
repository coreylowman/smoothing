from algorithms import *
from terminators import *
from functions import *
from smoothing import *

args_by_name = {
    'none': {'sample_fn': None, 'size': 0.5, 'num_points': 10, 'reduction_frequency': 80, 'reduction_pct': 0.3,
             'on': False},
    'F_G': {'sample_fn': gaussian_sample, 'size': 3.0, 'num_points': 10, 'reduction_frequency': 80,
            'reduction_pct': 0.3},
    'F_{S1}': {'sample_fn': sphere_sample, 'size': 3.0, 'num_points': 10, 'reduction_frequency': 80,
               'reduction_pct': 0.3},
    'F_{S2}': {'sample_fn': sphere_sample, 'size': 1.0, 'num_points': 10, 'reduction_frequency': 50,
               'reduction_pct': 1.0},
    'F_{S3}': {'sample_fn': sphere_sample, 'size': 1.0, 'num_points': 10, 'reduction_frequency': 80,
               'reduction_pct': 0.3},
}

algs_by_name = {
    # 'GA': genetic_algorithm,
    # 'ES': evolution_strategy,
    # 'DE': differential_evolution,
}

smoother_name = 'none'

for f in [
    YangFlockton.f1,
    YangFlockton.f2,
]:
    mins, maxs = get_bounds(f)
    f_min = get_f_min(f)

    print(f.__name__, f_min)

    for alg_name, alg in algs_by_name.items():
        for i in range(30):
            term = Any(ConvergenceTerminator(100, 0.001), GenerationTerminator(2000))
            smoother = Smoother(terminate_fn=term, fitness_fn=f, **args_by_name[smoother_name])
            fitness_by_individual = alg(smoother.terminate, smoother, mins, maxs,
                                        step_size_fn=smoother.step_size, observe_fn=smoother.observe)

            population = list(fitness_by_individual.keys())
            best_fitness = min(map(f, population))

            results = [smoother_name, f.__name__, alg_name, smoother.function_evaluations, smoother.total_generations,
                       best_fitness]
            with open('results.csv', 'a') as fp:
                fp.write(','.join(list(map(str, results))) + '\n')
            print(i, results)
