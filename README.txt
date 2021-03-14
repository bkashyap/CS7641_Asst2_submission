Assignment 2: Randomized Optimization

Link to code and data https://github.com/bkashyap/CS7641_Asst2_submission

All analysis is done in Jupyter IPython Notebooks you should be able to see the charts and code by just opening the files in github. Here are the direct links
Discrete Optimization Problems
1. ContinuousPeaks Problem Analysis - https://github.com/bkashyap/CS7641_Asst2_submission/blob/master/ContinuousPeaks.ipynb
2. Knapsack Problem Analysis - https://github.com/bkashyap/CS7641_Asst2_submission/blob/master/Knapsack.ipynb
3. TravelingSalesman Problem Analysis - https://github.com/bkashyap/CS7641_Asst2_submission/blob/master/TravelingSalesman.ipynb
Neural Networks
Neural Network Complete Analysis - https://github.com/bkashyap/CS7641_Asst2_submission/blob/master/NeuralNetwork.ipynb


How to run Instructions

Discrete Optimization
For Discrete Optimization I use ABAGAIL Library.
CSV files containing data for analysis are generated from code in ABAGAIL/jython folder, these are then read in through the following Jupyter Notebooks to do analysis
1. ContinuousPeaks.ipynb - Contains all code for generating charts for Continuous Peaks (CP) Analysis
2. Knapsack.ipynb - Contains all code for generating charts for Knapsack (KP) Analysis
3. TravelingSalesman-final.ipynb - Contains all code for generating charts for Traveling Salesman Problem (TSP) Analysis

The following Files are used for generating the CSV.
1. ABAGAIL/jython/continuouspeaks_asst2.py - for CP
2. ABAGAIL/jython/knapsack_asst2.py - for KP
3. ABAGAIL/jython/travelingsalesman_asst2.py - for TSP
4. ABAGAIL/jython/run.sh is used for running above files

Instructions for running Jupyter Notebooks for analysis of both Discrete Optimization and Neural Network Sections:
1. Sync code from github repository from https://github.com/bkashyap/CS7641_Asst2_submission
2. requirements.txt contains all dependencies (The major ones are pandas, numpy, jupyter, seaborn, matplotlib). Create Python3 environment using requirements.txt by running "pip3 install -r requirements.txt" while inside project directory
3. Open localhost:8888 on your browser and navigate and open ContinuousPeaks.ipynb, Knapsack.ipynb, TravelingSalesman.ipynb or NeuralNetwork.ipynb
4. If you need to run the code you can run individual code blocks by pressing shift + enter

Instructions for running Jython code to generate raw CSV files:
1. Install Jython
2. Run "sudo ./ABAGAIL/jython/run.sh" to run tuned ROs and return average of 10 rounds
3. For running other experiments all experiments have been commented clearly within the jython files mentioned above. Feel free to uncomment them to run them. I have commented them out by default since it would take hours to run if all are uncommented at once.
