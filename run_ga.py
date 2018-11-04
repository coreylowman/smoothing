from typing import List
from algorithms import genetic_algorithm, Individual
from terminators import *
from observers import *
from functions import *

terminator = NoImprovementsTerminator(10, 0.01)

for f in [YaoLiuLin.f1]:
    mins, maxs = get_bounds(f)

    genetic_algorithm(terminator, BestIndividualPrinter(), f, mins, maxs, 40)
