# D-SHIELD Planner #

This directory contains Python code for running the D-SHIELD Obervation and Downlink Planner. 

#### Requirements:
You need to have a MILP solver installed which can be either SCIP or Gurobi. SCIP is free. 

It's easy to install SCIP's Python API using pip:
'> pip install pyscipopt'

Here's the github link SCIP's Python API which has all the docs.
https://github.com/scipopt/PySCIPOpt

You also need to install gurobi py even if you don't use it (or have a license for it). 
'> pip install gurobipy'


### How to run the D-SHIELD Planner ###

1. Make sure you have downloaded the d-shield demo data to your computer. You can find it here with a README file.
https://github.com/dshield-proj/dshield-2026-demo/tree/main


2. To run the planner, run the Python program dshieldPlanner.py with two command line parameters: demo_data_directory and plan_creation_date.

The first parameter (demo_data_directory) is the filepath the the D-SHIELD demo data you downloaded in step 1. 
The second parameter is a date in YYYMMDD format corresponding to the 'plan_creation_date', which is demo date (between 7/1/26 and 7/10/26) for which this plan is being created. 

For example specifying '20260703' will use the planner inputs corresponding to demo date 7/3/26, and create a plan to be executed the following date 7/4/26, called the 'plan_execution_date', which is always the day after the 'plan_creation_date'.

Here is an example how how to run the planner from terminal command line with the two parameters

% python dshieldPlanner.py '/Users/Applications/dshield-2026-demo/' '20260703'



3. Output: The output will appear in the dshield-demo-data-2026/planner/output/ directory. There will be a subfolder with the name of the plan_execution_date. If the input parameter for the plan_creation_date was '20260703' then the output will appear in 'dshield-demo-data-2026/planner/output/20260704'.  