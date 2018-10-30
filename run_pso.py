from typing import List, Tuple
from algorithms import particle_swarm_optimization, Individual
from terminators import *
from observers import *
from functions import *

terminator = NoImprovementsTerminator(10, 0.01)

for f in [YaoLiuLin.f2]:
    mins, maxs = get_bounds(f)
    #
    particle_swarm_optimization(terminator, Population3DPlotter(f, mins[0], maxs[0]), f, mins, maxs)
