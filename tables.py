from collections import defaultdict
from statistics import mean, stdev

evals_by_fn_and_alg = defaultdict(list)
gens_by_fn_and_alg = defaultdict(list)
fmins_by_fn_and_alg = defaultdict(list)

algs = ['GA', 'SS(GA)', 'ES', 'SS(ES)', 'DE', 'SS(DE)', 'PSO', 'SS(PSO)']
fn1s = ['f1', 'F_G(f1)', 'F_{S5}(f1)', 'F_{S2}(f1)']
fn2s = ['f2', 'F_G(f2)', 'F_{S5}(f2)', 'F_{S2}(f2)']

table_idx = 1

with open('results.csv') as fp:
    for line in fp.readlines():
        smoother_name, fn, alg, evals, gens, fmins = line.strip().split(',')
        name = '{}({})'.format(smoother_name, fn) if smoother_name != 'none' else fn
        evals_by_fn_and_alg[(name, alg)].append(int(evals))
        gens_by_fn_and_alg[(name, alg)].append(int(gens))
        fmins_by_fn_and_alg[(name, alg)].append(float(fmins))


def wrap_fn_name(name):
    return '$' + name.replace('f', 'f_') + '$'


def output_table_for(data_by_fn_and_alg, fmt='{mean:.4f}\\pm{stdev:.3f}'):
    global table_idx
    for fns in [fn1s, fn2s]:
        print('\\begin{table}[h!]')
        print('\\centering')
        print('\\caption{FILL ME IN}')
        print('\\begin{tabular}{|c||c|c|c|c|}')
        print('\\hline')
        print(' &' + ' & '.join(map(wrap_fn_name, fns)) + '\\\\')
        print('\\hline\\hline')
        for alg in algs:
            means = []
            stdevs = []
            for fn in fns:
                data = data_by_fn_and_alg[(fn, alg)]
                means.append(mean(data))
                stdevs.append(stdev(data))

            results = []
            min_mean = min(means)
            min_mod_mean = min(means[1:])
            for i in range(len(means)):
                res = fmt.format(mean=means[i], stdev=stdevs[i])
                if means[i] == min_mean:
                    res = '\\mathbf{' + res + '}'
                if means[i] == min_mod_mean:
                    res = '\\underline{' + res + '}'
                results.append('$' + res + '$')

            print('{} & {} \\\\'.format(alg, ' & '.join(results)))
        print('\\hline')
        print('\\end{tabular}')
        print('\\label{table:' + str(table_idx) + '}')
        print('\\end{table}')
        table_idx += 1


output_table_for(fmins_by_fn_and_alg)
output_table_for(evals_by_fn_and_alg, fmt='{mean:.0f}\\pm{stdev:.0f}')
output_table_for(gens_by_fn_and_alg, fmt='{mean:.0f}\\pm{stdev:.0f}')
