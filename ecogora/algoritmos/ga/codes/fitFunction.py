import random
from deap import benchmarks

def fitnessFunction(x, parameters):
    globalOP = 0
    fitInd = 0
    if(parameters["BENCHMARK"] == "H1"):
        globalOP = benchmarks.h1([8.6998, 6.7665])[0]
        fitInd = benchmarks.h1(x)[0]
    elif(parameters["BENCHMARK"] == "BOHACHEVSKY"):
        globalOP = benchmarks.bohachevsky([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.bohachevsky(x)[0]
    elif(parameters["BENCHMARK"] == "HIMMELBLAU"):
        globalOP = benchmarks.himmelblau([3.0, 2.0])[0]
        fitInd = benchmarks.himmelblau(x)[0]
    elif(parameters["BENCHMARK"] == "SPHERE"):
        globalOP = benchmarks.sphere([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.sphere(x)[0]
    elif(parameters["BENCHMARK"] == "ROSENBROCK"):
        globalOP = benchmarks.rosenbrock([1 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.rosenbrock(x)[0]
    elif(parameters["BENCHMARK"] == "SCHAFFER"):
        globalOP = benchmarks.schaffer([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.schaffer(x)[0]
    elif(parameters["BENCHMARK"] == "SCHWEFEL"):
        globalOP = benchmarks.schwefel([420.9687436 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.schwefel(x)[0]
    elif(parameters["BENCHMARK"] == "RASTRIGIN"):
        globalOP = benchmarks.rastrigin([0 for _ in range(parameters["NDIM"])])[0]
        fitInd = benchmarks.rastrigin(x)[0]


    fitness = abs(fitInd - globalOP )
    return fitness
