#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#

export CLASSPATH=ABAGAIL.jar:$CLASSPATH
mkdir -p data/plot logs image

# Uncomment the one you want to run

# continuous peaks
#echo "continuous peaks"
# jython continuouspeaks_asst2.py

# Knapsack Problem
#echo "knapsack"
#jython knapsack_asst2.py

# Traveling Salesman Problem
#echo "TSP"
#jython travelingsalesman_asst2.py
