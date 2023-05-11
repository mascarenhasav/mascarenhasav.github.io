import random
from deap import benchmarks

def fitnessFunction(x, parameters):
    globalOP = 0
    fitInd = 0
    if(parameters["BENCHMARK"] == "H1"):
        globalOP = benchmarks.h1([8.6998, 6.7665])[0]
        fitInd = benchmarks.h1(x)[0]
    elif(parameters["BENCHMARK"] == "BOHA"):
        globalOP = benchmarks.bohachevsky([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.bohachevsky(x)[0]
    elif(parameters["BENCHMARK"] == "HIMME"):
        globalOP = benchmarks.himmelblau([3.0, 2.0])[0]
        fitInd = benchmarks.himmelblau(x)[0]
    elif(parameters["BENCHMARK"] == "SPHERE"):
        globalOP = benchmarks.sphere([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.sphere(x)[0]

    fitness = abs(fitInd - globalOP )
    return fitness
