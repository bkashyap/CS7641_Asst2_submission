import sys
import os
import time
import csv

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import util.ABAGAILArrays as ABAGAILArrays

#548415
Distribution.random.setSeed(548415)
ABAGAILArrays.random.setSeed(1000)
"""
Commandline parameter(s):
   none
"""


def train(alg_func, alg_name, ef, iters, label, expt):
    # Distribution.random.setSeed(548415)
    # ABAGAILArrays.random.setSeed(1000)
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = expt + "-" + alg_name + "-" + label + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "continous-peaks", FILE_NAME)
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

N=150
T=N/50
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# Try 5 rounds of RHC
# expt = "expt_Restarts"
# for i in range(5):
#     rhc = RandomizedHillClimbing(hcp)
#     train(rhc, "RHC", ef, 200000, "round=" + str(i), expt)
#     print "RHC Round" + str(i) + ": " + str(ef.value(rhc.getOptimal()))

#SA Tuning
# Expt1  - Tuning Temp
# expt = "expt_Temp"
# sa = SimulatedAnnealing(1E11, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E11_Decay=0.95",expt)
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E7, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E7_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E5, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E3, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E3_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E1, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E1_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E0, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E0_Decay=0.95", expt)

# Expt2  - Tuning Decay
# expt = "expt_Decay"
# sa = SimulatedAnnealing(1E5, .9999, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.9999", expt)
# sa = SimulatedAnnealing(1E5, .999, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.999", expt)
# sa = SimulatedAnnealing(1E5, .99, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.99", expt)
# sa = SimulatedAnnealing(1E5, .98, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.98", expt)
# sa = SimulatedAnnealing(1E5, .97, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.97", expt)
# sa = SimulatedAnnealing(1E5, .96, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.96", expt)
# sa = SimulatedAnnealing(1E5, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E5, .90, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.90", expt)
# sa = SimulatedAnnealing(1E5, .85, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E5_Decay=0.85", expt)

# GA Tuning
# Expt  - GA Population , select 750
#
# def GA_pop_size(pops):
#     mate_size = 50
#     mutation = 10
#     expt = "expt_GA_pop" + "_mate_size=" + str(mate_size) + "_mutation=" + str(mutation)
#     iterations = 40000
#
#     for pop in pops:
#         ga = StandardGeneticAlgorithm(pop, mate_size, mutation, gap)
#         train(ga, "GA", ef, iterations, "Pop=" + str(pop), expt)
#
# GA_pop_size([100,125,200,250,300, 350])


# def GA_mate_size(mates):
#     pop_size = 200
#     mutation = 10
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mutation=" + str(mutation)
#     iterations = 40000
#
#     for mate in mates:
#         ga = StandardGeneticAlgorithm(pop_size, mate, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mate=" + str(mate), expt)
#
#
# GA_mate_size([20,40,50, 60, 80,100])

# Expt  - GA mutate , select 750

# def GA_mutation_size(mutations):
#     pop_size = 200
#     mate_pop = 50
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mate=" + str(mate_pop)
#     iterations = 40000
#
#     for mutation in mutations:
#         ga = StandardGeneticAlgorithm(pop_size, mate_pop, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mutation=" + str(mutation), expt)
#
#
# GA_mutation_size([5,10,15,25,50])


# # # Expt  - GA Population
# expt = "expt_GA_pop"
# #
# ga = StandardGeneticAlgorithm(100, 100, 10, gap)
# train(ga, "GA", ef, 100000, "Pop=100", expt)
# ga = StandardGeneticAlgorithm(150, 100, 10, gap)
# train(ga, "GA", ef, 100000, "Pop=150", expt)
# ga = StandardGeneticAlgorithm(200, 100, 10, gap)
# train(ga, "GA", ef, 100000, "Pop=200", expt)
# ga = StandardGeneticAlgorithm(300, 100, 10, gap)
# train(ga, "GA", ef, 100000, "Pop=300", expt)

# # Expt  - GA mate
# expt = "expt_GA_mate"
#
# ga = StandardGeneticAlgorithm(200, 25, 10, gap)
# train(ga, "GA", ef, 40000, "mate=25", expt)
# ga = StandardGeneticAlgorithm(200, 50, 10, gap)
# train(ga, "GA", ef, 40000, "mate=50", expt)
# ga = StandardGeneticAlgorithm(200, 100, 10, gap)
# train(ga, "GA", ef, 40000, "mate=100", expt)
# ga = StandardGeneticAlgorithm(200, 125, 10, gap)
# train(ga, "GA", ef, 40000, "mate=125", expt)

# Expt  - GA mate
# expt = "expt_GA_mutate"
#
# ga = StandardGeneticAlgorithm(200, 50, 5, gap)
# train(ga, "GA", ef, 40000, "mutate=5", expt)
# ga = StandardGeneticAlgorithm(200, 50, 10, gap)
# train(ga, "GA", ef, 40000, "mutate=10", expt)
# ga = StandardGeneticAlgorithm(200, 50, 15, gap)
# train(ga, "GA", ef, 40000, "mutate=15", expt)
# ga = StandardGeneticAlgorithm(200, 50, 20, gap)
# train(ga, "GA", ef, 40000, "mutate=20", expt)


# MIMIC Tuning
#Expt  - MIMIC Population
# expt = "expt_MIMIC_pop_2"
# #
#
# mimic = MIMIC(50, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "Pop=50", expt)
# mimic = MIMIC(100, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "Pop=100", expt)
# mimic = MIMIC(150, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "Pop=150", expt)
# mimic = MIMIC(200, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "Pop=200", expt)
# mimic = MIMIC(300, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "Pop=300", expt)

# Expt  - MIMIC Keep
# expt = "expt_MIMIC_keep"
# #
#
# mimic = MIMIC(200, 5, pop)
# train(mimic,"MIMIC", ef, 2000, "keep=5", expt)
# mimic = MIMIC(200, 10, pop)
# train(mimic,"MIMIC", ef, 2000, "keep=10", expt)
# mimic = MIMIC(200, 15, pop)
# train(mimic,"MIMIC", ef, 2000, "keep=15", expt)
# mimic = MIMIC(200, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "keep=20", expt)
# mimic = MIMIC(200, 25, pop)
# train(mimic,"MIMIC", ef, 2000, "keep=25", expt)


# # Expt Default values
# expt = "expt_def"
#
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "def", expt)
# #print "RHC: " + str(ef.value(rhc.getOptimal()))
#
# sa = SimulatedAnnealing(1E5, .98, hcp)
# train(sa, "SA", ef, 200000, "def", expt)
# #print "SA: " + str(ef.value(sa.getOptimal()))
#
# ga = StandardGeneticAlgorithm(200, 100, 10, gap)
# train(ga, "GA", ef, 4000, "def", expt)
# #print "GA: " + str(ef.value(ga.getOptimal()))
#
# mimic = MIMIC(200, 20, pop)
# train(mimic,"MIMIC", ef, 2000, "def", expt)



#Expt Tuned values, size N =150
#expt = "expt_final_2"

# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
# #print "RHC: " + str(ef.value(rhc.getOptimal()))
#
# sa = SimulatedAnnealing(1E5, .98, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
# #print "SA: " + str(ef.value(sa.getOptimal()))

# ga = StandardGeneticAlgorithm(200, 60, 10, gap)
# train(ga, "GA", ef, 40000, "tuned", expt)
# print "GA: " + str(ef.value(ga.getOptimal()))

# mimic = MIMIC(200, 10, pop)
# train(mimic,"MIMIC", ef, 2000, "tuned", expt)


# Expt Tuned values, size N =250

# N=250
# T=N/50
# fill = [2] * N
# ranges = array('i', fill)
#
# ef = ContinuousPeaksEvaluationFunction(T)
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = SingleCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
#
# expt = "expt_tuned_big_final"
#
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
# #print "RHC: " + str(ef.value(rhc.getOptimal()))
#
# sa = SimulatedAnnealing(1E5, .98, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
# #print "SA: " + str(ef.value(sa.getOptimal()))
#
# ga = StandardGeneticAlgorithm(200, 50, 10, gap)
# train(ga, "GA", ef, 40000, "tuned", expt)
# #print "GA: " + str(ef.value(ga.getOptimal()))
#
# mimic = MIMIC(200, 10, pop)
# train(mimic,"MIMIC", ef, 2000, "tuned", expt)


# Expt Tuned values, size N =250

# N=150
# T=N/50
# fill = [2] * N
# ranges = array('i', fill)
#
# ef = ContinuousPeaksEvaluationFunction(T)
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = SingleCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
#
# expt = "expt_tuned_small"
#
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
# #print "RHC: " + str(ef.value(rhc.getOptimal()))
#
# sa = SimulatedAnnealing(1E5, .98, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
# #print "SA: " + str(ef.value(sa.getOptimal()))
#
# ga = StandardGeneticAlgorithm(200, 50, 15, gap)
# train(ga, "GA", ef, 4000, "tuned", expt)
# #print "GA: " + str(ef.value(ga.getOptimal()))
#
# mimic = MIMIC(200, 10, pop)
# train(mimic,"MIMIC", ef, 2000, "tuned", expt)

# Small Size
# Final averaged results
# RHC= 95.2
# SA= 98.0
# GA= 97.5
# MIMIC= 98.0

# Medium Size
# Final averaged results
# RHC= 282.4
# SA= 296.0
# GA= 220.8
# MIMIC= 294.3

# Large size
# Final averaged results
# RHC= 420.5
# SA= 494.0
# GA= 290.4
# MIMIC= 471.75

expt = "expt_avg"

score_RHC, score_SA, score_GA, score_MIMIC = [],[],[],[]

for i in range(10):
    N = 150
    T = N / 50
    fill = [2] * N
    ranges = array('i', fill)

    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    expt = "expt_avg"

    rhc = RandomizedHillClimbing(hcp)
    score_RHC.append(train(rhc, "RHC", ef, 200000, "tuned", expt))
    #print "RHC: " + str(ef.value(rhc.getOptimal()))

    sa = SimulatedAnnealing(1E5, .98, hcp)
    score_SA.append(train(sa, "SA", ef, 200000, "tuned", expt))
    #print "SA: " + str(ef.value(sa.getOptimal()))

    ga = StandardGeneticAlgorithm(200, 50, 15, gap)
    score_GA.append(train(ga, "GA", ef, 4000, "tuned", expt))
    #print "GA: " + str(ef.value(ga.getOptimal()))

    mimic = MIMIC(200, 10, pop)
    score_MIMIC.append(train(mimic,"MIMIC", ef, 2000, "tuned", expt))


print("Final averaged results")
print("RHC= " + str(sum(score_RHC)/len(score_RHC)))
print("SA= " + str(sum(score_SA)/len(score_SA)))
print("GA= " + str(sum(score_GA)/len(score_GA)))
print("MIMIC= " + str(sum(score_MIMIC)/len(score_MIMIC)))
