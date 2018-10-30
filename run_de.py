from typing import List, Tuple
from algorithms import differential_evolution, Individual
from terminators import *
from observers import *
from functions import *

terminator = NoImprovementsTerminator(50, 0.01)

for f in [YaoLiuLin.f2]:
    mins, maxs = get_bounds(f)
    differential_evolution(terminator, BestIndividualPrinter(), f, mins, maxs, 20, 0.1, 0.5)
