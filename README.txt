Assignment 2: Randomized Optimization

Link to code and data

All analysis is done in Jupyter IPython Notebooks you should be able to see the charts and code by just opening the files in github. Here are the direct links
1. Wine Quality - https://github.com/bkashyap/CS7641_Asst1_submission/blob/master/WineQuality.ipynb
2. Breast cancer - https://github.com/bkashyap/CS7641_Asst1_submission/blob/master/BreastCancer.ipynb

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

Instructions for running:
1. Sync code from github repository
2. Select


Neural Networks




Code used from external sources have been referenced in the comments.

Here is the link to the entire repository containing the data and notebooks: https://github.com/bkashyap/CS7641_Asst1_submission

How to run:
1. Clone repository from github:https://github.com/bkashyap/CS7641_Asst1_submission
2. requirements.txt contains all dependencies (The major ones are pandas, numpy, jupyter, seaborn, matplotlib). Create Python3 environment using requirements.txt by running "pip3 install -r requirements.txt" while inside project directory
3. Run jupyter-notebook on a terminal
4. Open localhost:8888 on your browser and navigate and open WineQuality.ipynb or BreastCancer.ipynb
5. If you need to run the code you can run individual code blocks by pressing shift + enter
