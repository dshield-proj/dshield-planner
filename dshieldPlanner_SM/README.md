# D-SHIELD Planner for Soil Moisture use case#

This directory contains Python code for running the D-SHIELD Obervation and Downlink Planner. 

#### Requirements:
You need to have a MILP solver installed which can be either SCIP or Gurobi. SCIP is free. 

It's easy to install SCIP's Python API using pip:
'> pip install pyscipopt'

Here's the github link SCIP's Python API which has all the docs.
https://github.com/scipopt/PySCIPOpt

You also need to install gurobi py even if you don't use it (or have a license for it). 
'> pip install gurobipy'


### How to run the D-SHIELD Planner for soil moisture ###

1. Make sure you have downloaded the d-shield soil moisture sample data and place it in the directory /dshieldPlanner_SM/inputs. (contact us for a link to that data). 

2. cd to this directory /dshieldPlanner_SM/

3. To run the planner, run the Python program dshieldPlanner_SM.py. 

Here is an example how how to run the planner from terminal command line:

% python  dshieldPlanner_SM.py
