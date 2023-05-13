'''
Apply LS on the best
'''
def localSearch(best, toolbox, parameters, fitFunction):
    rls  = parameters["RLS"]
    bp = creator.Particle(best)
    for _ in range(parameters["ETRY"]):
        for i in range(parameters["NDIM"]):
            bp[i] = bp[i] + random.uniform(-1, 1)*rls
        bp.fitness.values = evaluate(bp, fitFunction, parameters=parameters)
        if bp.fitness > best.fitness:
            best = creator.Particle(bp)
            best.fitness.values = bp.fitness.values
    return best

