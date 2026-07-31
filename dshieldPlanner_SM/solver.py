import os
import gurobipy as gp
from gurobipy import GRB

from pyscipopt import Model

class Solver:
    def __init__ (self, type):
        self.type = type  # gurobi or scip
        self.model = None
        print("Solver() "+str(self.type))

    def isGurobi(self):
         return self.type == "gurobi"

    def initSolver(self, mipGap=None, timeLimit=None):
        print("initSolver()")
        if self.isGurobi():
            gurobiLogFile = "gurobiLog.txt"
            if os.path.exists(gurobiLogFile):
                os.remove(gurobiLogFile)
            self.model = gp.Model()
            self.model.Params.logFile = "gurobiLog.txt"
            self.model.Params.Method = 2
            if mipGap:
                self.model.Params.MIPGap = mipGap
            if timeLimit:
                self.model.Params.TimeLimit = timeLimit # seconds
            return self.model
        else:
            self.model = Model()  # the name is optional
            if timeLimit:
                self.model.setRealParam("limits/time", timeLimit) # seconds
            if mipGap:
                self.model.setRealParam('limits/gap', mipGap)
            self.model.writeParams()

    def addIntegerVars(self, indices, name, lb, ub):
        vars = {}
        if self.isGurobi():
            tupleDictVars = self.model.addVars(indices, vtype=GRB.INTEGER, name=name, lb=lb, ub=ub)
            vars = dict(tupleDictVars) # convert from tupledict to dict
            return vars
        else:
            for i in indices:
                var = self.model.addVar(name=name+"["+str(i)+"]", vtype='I', lb=lb, ub=ub)
                # print(str(var))
                vars.update({i:var})
            # print("vars: "+str(vars))
        return vars

    def addContinuousVar(self, name, lb, ub):
        assert self.isGurobi(), "ERROR: solver.addContinousVar() not implemented for SCIP"
        if self.isGurobi():
            tupleDictVar = self.model.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name=name)
            return tupleDictVar

    def addContinuousVars(self, indices, name, lb, ub):
        vars = {}
        if self.isGurobi():
            tupleDictVars = self.model.addVars(indices, vtype=GRB.CONTINUOUS, name=name, lb=lb, ub=ub)
            vars = dict(tupleDictVars) # convert from tupledict to dict
            return vars
        else:
            for i in indices:
                var = self.model.addVar(name=name+"["+str(i)+"]", vtype='C', lb=lb, ub=ub)
                # print(str(var))
                vars.update({i:var})
            # print("vars: "+str(vars))
        return vars

    def addContinuousObjectiveVars(self, indices, obj, name, lb, ub):
        vars = {}
        if self.isGurobi():
            tupleDictVars = self.model.addVars(indices, obj= obj, vtype=GRB.CONTINUOUS, name=name, lb=lb, ub=ub)
            vars = dict(tupleDictVars) # convert from tupledict to dict
            return vars
        else:
            for i in indices:
                var = self.model.addVar(name=name+"["+str(i)+"]", vtype='C', lb=lb, ub=ub)
                # print(str(var))
                vars.update({i:var})
            # print("vars: "+str(vars))
        return vars

    def addBinaryVars(self, indices, name):
        vars = {}
        if self.isGurobi():
            tupleDictVars = self.model.addVars(indices, vtype=GRB.BINARY, name=name)
            vars = dict(tupleDictVars) # convert from tupledict to dict
        else:
            for i in indices:
                var = self.model.addVar(name=name+str(i), vtype='B')
                # print(str(var))
                vars.update({i:var})
            # print("vars: "+str(vars))
        return vars

    def addBinaryObjectiveVars(self, indices, obj, name):
        vars = {}
        if self.isGurobi():
            tupleDictVars = self.model.addVars(indices, obj=obj, vtype=GRB.BINARY, name=name)
            vars = dict(tupleDictVars) # convert from tupledict to dict
        else:
            n = 0
            for i in indices:
                var = self.model.addVar(name=name+str(i), obj= obj[n], vtype='B')
                # print(str(var))
                vars.update({i:var})
                n += 1
            # print("vars: "+str(vars))
        return vars

    def setObjectiveSense(self, sense):
        if self.isGurobi():
            val = gp.GRB.MAXIMIZE if sense == "maximize" else gp.GRB.MINIMIZE
            self.model.setAttr(gp.GRB.Attr.ModelSense, val)
        else:
            self.model.setMaximize()

    def addConstraint(self, constraint, name):
        if self.isGurobi():
            self.model.addConstr(constraint, name)
        else:
            self.model.addCons(constraint, name)

    def solveIt(self):
        print("solveIt()")
        if self.isGurobi():
            self.model.optimize()
            if self.model.Status != GRB.INFEASIBLE:
                self.model.write("solution.sol")
                return True
            else:
                self.model.computeIIS()
                self.model.write('iismodel.ilp')
                return False
        else:
            self.model.optimize()
            status = self.model.getStatus()
            if status == "infeasible":
                return False
            else:
                return True

    def getVarValue(self, var):
        if self.isGurobi():
            return getattr(var, 'x')
        else:
            return self.model.getVal(var)



    def extractSelectedBinaryVars(self, vars):
        print("extractSelectedBinaryVars()")
        selected = []
        for key in vars:
            var = vars[key]
            val = self.getVarValue(var)
            if val >= 0.5:
                selected.append(key)
        return selected

    def extractUnselectedBinaryVars(self, vars):
        print("extractUnselectedBinaryVars()")
        selected = []
        for key in vars:
            var = vars[key]
            val = self.getVarValue(var)
            if val < 0.5:
                selected.append(key)
        return selected

    def tuneModel(self, timeLimit):
        print("\ntuneModel() timeLimit: "+str(timeLimit/3600) + " hrs")
        self.model.Params.TuneTimeLimit = timeLimit
        self.model.Params.TuneOutput = 1
        self.model.Params.TuneCriterion = 2
        self.model.tune()
        for i in range(self.model.tuneResultCount):
            self.model.getTuneResult(i)
            self.model.write('tune' + str(i) + '.prm')
        print("\nTuning Complete!")


    def writeModel(self, filename):
        if self.isGurobi():
            self.model.write(filename+".lp")
        else:
            self.model.writeProblem()
