# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes
# to a file and plot them in your favorite tool.
import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import csv
import util.ABAGAILArrays as ABAGAILArrays

Distribution.random.setSeed(548415)
ABAGAILArrays.random.setSeed(1000)

from array import array

def train(alg_func, alg_name, ef, iters, label, expt, output= True):
    Distribution.random.setSeed(548415)
    ABAGAILArrays.random.setSeed(1000)

    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = expt + "-" + alg_name + "-" + label + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "tsp", FILE_NAME)
    print(OUTPUT_FILE)

    with open(OUTPUT_FILE, "wb") as results:
        writer = csv.writer(results, delimiter=',')
        writer.writerow(["iters", "fevals", "fitness", "time"])
        start = time.time()
        for i in range(iters):
            fit.train()
            end = time.time()
            writer.writerow([i, ef.getFunctionEvaluations() - i, ef.value(alg_func.getOptimal()), (end-start)])


    print alg_name + ": " + str(ef.value(alg_func.getOptimal()))
    print "Function Evaluations: " + str(ef.getFunctionEvaluations()-iters)
    print "Iters: " + str(iters)
    print "####"
    return ef.value(alg_func.getOptimal())


"""
Commandline parameter(s):
    none
"""

# set N value.  This is the number of points
N = 25
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# #Try 5 rounds of RHC
# expt = "expt_Restarts"
# for i in range(5):
#     rhc = RandomizedHillClimbing(hcp)
#     train(rhc, "RHC", ef, 20000, "round=" + str(i), expt)
#     print "RHC Round" + str(i) + ": " + str(ef.value(rhc.getOptimal()))



# expt = "expt_med"
#
#
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 20000, "test", expt)
# print "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(rhc.getOptimal().getDiscrete(x))
# print path
#
# sa = SimulatedAnnealing(1E9, .98, hcp)
# train(sa, "SA", ef, 24000, "test", expt)
# print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(sa.getOptimal().getDiscrete(x))
# print path
#
#
# ga = StandardGeneticAlgorithm(225, 40, 5, gap)
# train(ga, "GA", ef, 40000, "test", expt)
# print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))
# print "Route:"
# path = []
# for x in range(0,N):
#     path.append(ga.getOptimal().getDiscrete(x))
# print path
#
#
# #for mimic we use a sort encoding
# ef = TravelingSalesmanSortEvaluationFunction(points);
# fill = [N] * N
# ranges = array('i', fill)
# odd = DiscreteUniformDistribution(ranges);
# df = DiscreteDependencyTree(.1, ranges);
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df);
# #
# mimic = MIMIC(150, 20, pop)
# train(mimic,"MIMIC", ef, 4000, "test", expt)
# print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))
# print "Route:"
# path = []
# optimal = mimic.getOptimal()
# fill = [0] * optimal.size()
# ddata = array('d', fill)
# for i in range(0,len(ddata)):
#     ddata[i] = optimal.getContinuous(i)
# order = ABAGAILArrays.indices(optimal.size())
# ABAGAILArrays.quicksort(ddata, order)
# print order


# SA Tuning
#SA - 1E9, 0.95
#Expt1  - Tuning Temp, select 1E9

# decay = 0.98
# expt = "expt_Temp" + "_decay="+ str(decay)
#
# sa = SimulatedAnnealing(1E11, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E11",expt)
# sa = SimulatedAnnealing(1E9, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E9", expt)
# sa = SimulatedAnnealing(1E7, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E7", expt)
# sa = SimulatedAnnealing(1E5, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E5", expt)
# sa = SimulatedAnnealing(1E3, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E3", expt)
# sa = SimulatedAnnealing(1E1, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E1", expt)
# sa = SimulatedAnnealing(1E0, decay, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E0", expt)
#
# # Expt2  - Tuning Decay, select 0.95
#
# temp = 1E9
# expt = "expt_Decay" + "_temp="+ str(decay)
# sa = SimulatedAnnealing(temp, .9999, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.9999", expt)
# sa = SimulatedAnnealing(temp, .999, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.999", expt)
# sa = SimulatedAnnealing(temp, .99, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.99", expt)
# sa = SimulatedAnnealing(temp, .98, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.98", expt)
# sa = SimulatedAnnealing(temp, .97, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.97", expt)
# sa = SimulatedAnnealing(temp, .96, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.96", expt)
# sa = SimulatedAnnealing(temp, .95, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.95", expt)
# sa = SimulatedAnnealing(temp, .90, hcp)
# train(sa, "SA", ef, 30000, "Decay=0.90", expt)


# GA Tuning

# # Expt  - GA Population , select 750

# def GA_pop_size(pops):
#     mate_size = 40
#     mutation = 20
#     expt = "expt_GA_pop" + "_mate_size=" + str(mate_size) + "_mutation=" + str(mutation)
#     iterations = 10000
#
#     for pop in pops:
#         ga = StandardGeneticAlgorithm(pop, mate_size, mutation, gap)
#         train(ga, "GA", ef, iterations, "Pop=" + str(pop), expt)
#
# GA_pop_size([225, 250,275,300,325, 350])
#
# Expt  - GA mate , select 750


# def GA_mate_size(mates):
#     pop_size = 225
#     mutation = 5
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mutation=" + str(mutation)
#     iterations = 10000
#
#     for mate in mates:
#         ga = StandardGeneticAlgorithm(pop_size, mate, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mate=" + str(mate), expt)
#
#
# GA_mate_size([60, 80])


# Expt  - GA mutate , select 750

# def GA_mutation_size(mutations):
#     pop_size = 225
#     mate_pop = 40
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mate=" + str(mate_pop)
#     iterations = 10000
#
#     for mutation in mutations:
#         ga = StandardGeneticAlgorithm(pop_size, mate_pop, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mutation=" + str(mutation), expt)
#
#
# GA_mutation_size([7])

# Expt  - GA mutate, select 10

# pop_size = 1000
# mate = 100
# expt = "expt_GA_mutate" + "_pop_size="+ str(pop_size) + "_mate=" + str(mate)
#
# ga = StandardGeneticAlgorithm(pop_size, mate, 5, gap)
# train(ga, "GA", ef, 10000, "mutate=5", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 10, gap)
# train(ga, "GA", ef, 10000, "mutate=10", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 15, gap)
# train(ga, "GA", ef, 10000, "mutate=15", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 20, gap)
# train(ga, "GA", ef, 10000, "mutate=20", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 25, gap)
# train(ga, "GA", ef, 10000, "mutate=25", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 30, gap)
# train(ga, "GA", ef, 10000, "mutate=30", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 35, gap)
# train(ga, "GA", ef, 10000, "mutate=35", expt)
# ga = StandardGeneticAlgorithm(pop_size, mate, 40, gap)
# train(ga, "GA", ef, 10000, "mutate=40", expt)

# MIMIC Tuning

# Expt  - MIMIC Population
# keep = 20
# expt = "expt_MIMIC_pop" + "_keep="+ str(keep)
# #
#
# mimic = MIMIC(50, keep, pop)
# train(mimic,"MIMIC", ef, 1000, "Pop=50", expt)
# mimic = MIMIC(100, keep, pop)
# train(mimic,"MIMIC", ef, 1000, "Pop=100", expt)
# mimic = MIMIC(150, keep, pop)
# train(mimic,"MIMIC", ef, 1000, "Pop=150", expt)
# mimic = MIMIC(200, keep, pop)
# train(mimic,"MIMIC", ef, 1000, "Pop=200", expt)
# mimic = MIMIC(250, keep, pop)
# train(mimic,"MIMIC", ef, 1000, "Pop=250", expt)


# # Expt  - MIMIC Keep
# pop_size = 200
# expt = "expt_MIMIC_keep2" + "_pop_size="+ str(pop_size)
# # #
# #
# mimic = MIMIC(pop_size, 5, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=5", expt)
# mimic = MIMIC(pop_size, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=10", expt)
# mimic = MIMIC(pop_size, 15, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=15", expt)
# mimic = MIMIC(pop_size, 20, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=20", expt)
# mimic = MIMIC(pop_size, 25, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=25", expt)
# mimic = MIMIC(pop_size, 30, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=30", expt)
# mimic = MIMIC(pop_size, 35, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=35", expt)
# mimic = MIMIC(pop_size, 40, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=40", expt)


# set N value.  This is the number of points
# N=25 small
# RHC= 0.185613537863
# SA= 0.195824171227
# GA= 0.223269860735
# MIMIC= 0.1982821906

# N=50 medium
# Final averaged results
# RHC= 0.118872432255
# SA= 0.122783038838
# GA= 0.155069663407
# MIMIC= 0.12260121809

# N=75 medium
# RHC= 0.0894583884374
# SA= 0.0878659419958
# GA= 0.123354670439
# MIMIC= 0.0832376954058

expt = "expt_avg"

score_RHC, score_SA, score_GA, score_MIMIC = [],[],[],[]

for i in range(10):

    N = 50
    random = Random(1234 * i)

    points = [[0 for x in xrange(2)] for x in xrange(N)]
    for i in range(0, len(points)):
        points[i][0] = random.nextDouble()
        points[i][1] = random.nextDouble()

    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscretePermutationDistribution(N)
    nf = SwapNeighbor()
    mf = SwapMutation()
    cf = TravelingSalesmanCrossOver(ef)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

    expt = "expt_avg"

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 200000)
    score_RHC.append(train(rhc, "RHC", ef, 200000, "test", expt))
    print  "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))

    sa = SimulatedAnnealing(1E9, .98, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    score_SA.append(train(sa, "SA", ef, 200000, "test", expt))
    print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))

    ga = StandardGeneticAlgorithm(225, 40, 5, gap)
    fit = FixedIterationTrainer(ga, 1000)
    score_GA.append(train(ga, "GA", ef, 40000, "test", expt))
    print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))


    # for mimic we use a sort encoding
    ef = TravelingSalesmanSortEvaluationFunction(points);
    fill = [N] * N
    ranges = array('i', fill)
    odd = DiscreteUniformDistribution(ranges);
    df = DiscreteDependencyTree(.1, ranges);
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

    mimic = MIMIC(150, 20, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    score_MIMIC.append(train(mimic, "MIMIC", ef, 4000, "test", expt))
    print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))


print("Final averaged results")
print("RHC= " + str(sum(score_RHC)/len(score_RHC)))
print("SA= " + str(sum(score_SA)/len(score_SA)))
print("GA= " + str(sum(score_GA)/len(score_GA)))
print("MIMIC= " + str(sum(score_MIMIC)/len(score_MIMIC)))
