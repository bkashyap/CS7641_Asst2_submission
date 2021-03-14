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
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import csv
import util.ABAGAILArrays as ABAGAILArrays


def train(alg_func, alg_name, ef, iters, label, expt, output= True):
    # Distribution.random.setSeed(548415)
    # ABAGAILArrays.random.setSeed(1000)
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = expt + "-" + alg_name + "-" + label + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "knapsack", FILE_NAME)

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

# Random number generator */
random = Random(0)
# The number of items
NUM_ITEMS = 75
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


# #Try 5 rounds of RHC
# expt = "expt_Restarts"
# for i in range(5):
#     rhc = RandomizedHillClimbing(hcp)
#     train(rhc, "RHC", ef, 5000, "round=" + str(i), expt)
#     print "RHC Round" + str(i) + ": " + str(ef.value(rhc.getOptimal()))

#SA Tuning
# Expt1  - Tuning Temp, select 1E9
# expt = "expt_Temp_test2"
# sa = SimulatedAnnealing(1E11, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E11_Decay=0.95",expt)
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E9_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E7, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E7_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E5, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E5_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E3, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E3_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E1, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E1_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E0, .95, hcp)
# train(sa, "SA", ef, 100000, "Temp=1E0_Decay=0.95", expt)

# # Expt2  - Tuning Decay, select 0.95
# expt = "expt_Decay"
# sa = SimulatedAnnealing(1E9, .9999, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.9999", expt)
# sa = SimulatedAnnealing(1E9, .999, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.999", expt)
# sa = SimulatedAnnealing(1E9, .99, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.99", expt)
# sa = SimulatedAnnealing(1E9, .98, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.98", expt)
# sa = SimulatedAnnealing(1E9, .97, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.97", expt)
# sa = SimulatedAnnealing(1E9, .96, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.96", expt)
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.95", expt)
# sa = SimulatedAnnealing(1E9, .90, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.90", expt)
# sa = SimulatedAnnealing(1E9, .85, hcp)
# train(sa, "SA", ef, 30000, "Temp=1E9_Decay=0.85", expt)


# GA Tuning
# Expt  - GA Population , select 750

# def GA_pop_size(pops):
#     mate_size = 20
#     mutation = 5
#     expt = "expt_GA_pop" + "_mate_size=" + str(mate_size) + "_mutation=" + str(mutation)
#     iterations = 40000
#
#     for pop in pops:
#         ga = StandardGeneticAlgorithm(pop, mate_size, mutation, gap)
#         train(ga, "GA", ef, iterations, "Pop=" + str(pop), expt)
#
# GA_pop_size([100,125,200,250,300, 350])
#
#
# def GA_mate_size(mates):
#     pop_size = 300
#     mutation = 5
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mutation=" + str(mutation)
#     iterations = 40000
#
#     for mate in mates:
#         ga = StandardGeneticAlgorithm(pop_size, mate, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mate=" + str(mate), expt)
#
#
# GA_mate_size([20,40,60, 80,100])
#
# # Expt  - GA mutate , select 750
#
# def GA_mutation_size(mutations):
#     pop_size = 300
#     mate_pop = 5
#     expt = "expt_GA_mate" + "_pop_size=" + str(pop_size) + "_mate=" + str(mate_pop)
#     iterations = 40000
#
#     for mutation in mutations:
#         ga = StandardGeneticAlgorithm(pop_size, mate_pop, mutation, gap)
#         train(ga, "GA", ef, iterations, "Mutation=" + str(mutation), expt)
#
#
# GA_mutation_size([5,10,15,25,50])



# # # Expt  - GA Population , select 250
# expt = "expt_GA_pop"
# #
# ga = StandardGeneticAlgorithm(100, 100, 10, gap)
# train(ga, "GA", ef, 4000, "Pop=100", expt)
# ga = StandardGeneticAlgorithm(150, 100, 10, gap)
# train(ga, "GA", ef, 4000, "Pop=150", expt)
# ga = StandardGeneticAlgorithm(200, 100, 10, gap)
# train(ga, "GA", ef, 4000, "Pop=200", expt)
# ga = StandardGeneticAlgorithm(250, 100, 10, gap)
# train(ga, "GA", ef, 4000, "Pop=250", expt)

# # Expt  - GA mate , select 125
# expt = "expt_GA_mate"
#
# ga = StandardGeneticAlgorithm(250, 25, 10, gap)
# train(ga, "GA", ef, 4000, "mate=25", expt)
# ga = StandardGeneticAlgorithm(250, 50, 10, gap)
# train(ga, "GA", ef, 4000, "mate=50", expt)
# ga = StandardGeneticAlgorithm(250, 100, 10, gap)
# train(ga, "GA", ef, 4000, "mate=100", expt)
# ga = StandardGeneticAlgorithm(250, 125, 10, gap)
# train(ga, "GA", ef, 4000, "mate=125", expt)

# Expt  - GA mutate, select 10
# expt = "expt_GA_mutate"
#
# ga = StandardGeneticAlgorithm(250, 125, 5, gap)
# train(ga, "GA", ef, 4000, "mutate=5", expt)
# ga = StandardGeneticAlgorithm(250, 125, 10, gap)
# train(ga, "GA", ef, 4000, "mutate=10", expt)
# ga = StandardGeneticAlgorithm(250, 125, 15, gap)
# train(ga, "GA", ef, 4000, "mutate=15", expt)
# ga = StandardGeneticAlgorithm(250, 125, 20, gap)
# train(ga, "GA", ef, 4000, "mutate=20", expt)

#MIMIC Tuning
# Expt  - MIMIC Population
# expt = "expt_MIMIC_pop_new"
# #
#
# mimic = MIMIC(50, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=50", expt)
# mimic = MIMIC(100, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=100", expt)
# mimic = MIMIC(150, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=150", expt)
# mimic = MIMIC(200, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=200", expt)
# mimic = MIMIC(250, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=250", expt)
# mimic = MIMIC(300, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "Pop=300", expt)


# # Expt  - MIMIC Keep
# expt = "expt_MIMIC_keep"
# #
#
# mimic = MIMIC(250, 5, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=5", expt)
# mimic = MIMIC(250, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=10", expt)
# mimic = MIMIC(250, 15, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=15", expt)
# mimic = MIMIC(250, 20, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=20", expt)
# mimic = MIMIC(250, 25, pop)
# train(mimic,"MIMIC", ef, 4000, "keep=25", expt)
#
#
# expt = "expt_test_4"
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
#
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
#
# ga = StandardGeneticAlgorithm(300, 80, 5, gap)
# train(ga, "GA", ef, 40000, "tuned", expt)
#
# mimic = MIMIC(250, 10, pop)
# train(mimic,"MIMIC", ef, 4000, "tuned", expt)


# # Large size
# # Random number generator */
# random = Random(0)
# # The number of items
# NUM_ITEMS = 125
# # The number of copies each
# COPIES_EACH = 4
# # The maximum weight for a single element
# MAX_WEIGHT = 100
# # The maximum volume for a single element
# MAX_VOLUME = 50
# # The volume of the knapsack
# KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4
#
# # create copies
# fill = [COPIES_EACH] * NUM_ITEMS
# copies = array('i', fill)
#
# # create weights and volumes
# fill = [0] * NUM_ITEMS
# weights = array('d', fill)
# volumes = array('d', fill)
# for i in range(0, NUM_ITEMS):
#     weights[i] = random.nextDouble() * MAX_WEIGHT
#     volumes[i] = random.nextDouble() * MAX_VOLUME
#
#
# # create range
# fill = [COPIES_EACH + 1] * NUM_ITEMS
# ranges = array('i', fill)
#
# ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = UniformCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
# expt = "expt_tuned_large"
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
#
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
#
# ga = StandardGeneticAlgorithm(250, 125, 10, gap)
# train(ga, "GA", ef, 40000, "tuned", expt)
#
# mimic = MIMIC(250, 5, pop)
# train(mimic,"MIMIC", ef, 4000, "tuned", expt)
#
# Small size
# Random number generator */
# random = Random(0)
# # The number of items
# NUM_ITEMS = 25
# # The number of copies each
# COPIES_EACH = 4
# # The maximum weight for a single element
# MAX_WEIGHT = 15
# # The maximum volume for a single element
# MAX_VOLUME = 50
# # The volume of the knapsack
# KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4
#
# # create copies
# fill = [COPIES_EACH] * NUM_ITEMS
# copies = array('i', fill)
#
# # create weights and volumes
# fill = [0] * NUM_ITEMS
# weights = array('d', fill)
# volumes = array('d', fill)
# for i in range(0, NUM_ITEMS):
#     weights[i] = random.nextDouble() * MAX_WEIGHT
#     volumes[i] = random.nextDouble() * MAX_VOLUME
#
#
# # create range
# fill = [COPIES_EACH + 1] * NUM_ITEMS
# ranges = array('i', fill)
#
# ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = UniformCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
# expt = "expt_tuned_small"
# rhc = RandomizedHillClimbing(hcp)
# train(rhc, "RHC", ef, 200000, "tuned", expt)
#
# sa = SimulatedAnnealing(1E9, .95, hcp)
# train(sa, "SA", ef, 200000, "tuned", expt)
#
# ga = StandardGeneticAlgorithm(250, 125, 10, gap)
# train(ga, "GA", ef, 40000, "tuned", expt)
#
# mimic = MIMIC(250, 5, pop)
# train(mimic,"MIMIC", ef, 4000, "tuned", expt)

# N= 75
# Final averaged results
# RHC= 6289.98923334
# SA= 6332.00581176
# GA= 6946.37873683
# MIMIC= 7309.11278039
#

# expt = "expt_avg"
#
# score_RHC, score_SA, score_GA, score_MIMIC = [],[],[],[]
#
# for i in range(10):
#
#     # Random number generator change SEED every time*/
#     random = Random(1234*i)
#     # The number of items
#     NUM_ITEMS = 75
#     # The number of copies each
#     COPIES_EACH = 4
#     # The maximum weight for a single element
#     MAX_WEIGHT = 50
#     # The maximum volume for a single element
#     MAX_VOLUME = 50
#     # The volume of the knapsack
#     KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4
#
#     # create copies
#     fill = [COPIES_EACH] * NUM_ITEMS
#     copies = array('i', fill)
#
#     # create weights and volumes
#     fill = [0] * NUM_ITEMS
#     weights = array('d', fill)
#     volumes = array('d', fill)
#     for i in range(0, NUM_ITEMS):
#         weights[i] = random.nextDouble() * MAX_WEIGHT
#         volumes[i] = random.nextDouble() * MAX_VOLUME
#
#     # create range
#     fill = [COPIES_EACH + 1] * NUM_ITEMS
#     ranges = array('i', fill)
#
#     ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
#     odd = DiscreteUniformDistribution(ranges)
#     nf = DiscreteChangeOneNeighbor(ranges)
#     mf = DiscreteChangeOneMutation(ranges)
#     cf = UniformCrossOver()
#     df = DiscreteDependencyTree(.1, ranges)
#     hcp = GenericHillClimbingProblem(ef, odd, nf)
#     gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
#     pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
#     rhc = RandomizedHillClimbing(hcp)
#     score_RHC.append(train(rhc, "RHC", ef, 200000, "tuned", expt, False))
#
#     sa = SimulatedAnnealing(1E9, .95, hcp)
#     score_SA.append(train(sa, "SA", ef, 200000, "tuned", expt, False))
#
#     ga = StandardGeneticAlgorithm(250, 125, 10, gap)
#     score_GA.append(train(ga, "GA", ef, 40000, "tuned", expt, False))
#
#     mimic = MIMIC(250, 5, pop)
#     score_MIMIC.append(train(mimic, "MIMIC", ef, 4000, "tuned", expt, False))
#
# print("Final averaged results")
# print("RHC= " + str(sum(score_RHC)/len(score_RHC)))
# print("SA= " + str(sum(score_SA)/len(score_SA)))
# print("GA= " + str(sum(score_GA)/len(score_GA)))
# print("MIMIC= " + str(sum(score_MIMIC)/len(score_MIMIC)))



# set N value.  This is the number of points
# N=25 small, W = 15
# Final averaged results
# RHC= 594.985278218
# SA= 608.027783457
# GA= 686.980097436
# MIMIC= 689.381030792

# N=50 medium, W= 50
# Final averaged results
# RHC= 6930.17843209
# SA= 7087.4146633
# GA= 7744.91941157
# MIMIC= 8021.66013253

# N=75 large
# RHC= 24426.0050473
# SA= 24434.8201348
# GA= 26713.01408
# MIMIC= 28856.6496922

expt = "expt_avg"

score_RHC, score_SA, score_GA, score_MIMIC = [],[],[],[]

for i in range(10):

    # Random number generator */
    random = Random(1234*i)
    # The number of items
    NUM_ITEMS = 150
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 100
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    # create copies
    fill = [COPIES_EACH] * NUM_ITEMS
    copies = array('i', fill)

    # create weights and volumes
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME

    # create range
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)

    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    df = DiscreteDependencyTree(.1, ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
    expt = "expt_avg"

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 200000)
    score_RHC.append(train(rhc, "RHC", ef, 200000, "test", expt))
    print  "RHC Inverse of Distance: " + str(ef.value(rhc.getOptimal()))

    sa = SimulatedAnnealing(1E9, .95, hcp)
    score_SA.append(train(sa, "SA", ef, 200000, "test", expt))
    print "SA Inverse of Distance: " + str(ef.value(sa.getOptimal()))

    ga = StandardGeneticAlgorithm(300, 80, 5, gap)
    score_GA.append(train(ga, "GA", ef, 40000, "test", expt))
    print "GA Inverse of Distance: " + str(ef.value(ga.getOptimal()))

    mimic = MIMIC(250, 10, pop)
    score_MIMIC.append(train(mimic, "MIMIC", ef, 4000, "test", expt))
    print "MIMIC Inverse of Distance: " + str(ef.value(mimic.getOptimal()))


print("Final averaged results")
print("RHC= " + str(sum(score_RHC)/len(score_RHC)))
print("SA= " + str(sum(score_SA)/len(score_SA)))
print("GA= " + str(sum(score_GA)/len(score_GA)))
print("MIMIC= " + str(sum(score_MIMIC)/len(score_MIMIC)))
