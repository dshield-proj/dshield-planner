import os
import gurobipy as gp
from gurobipy import GRB

from pyscipopt import Model

class Solver:
    def __init__ (self, type):
        self.type = type  # gurobi or scip
        self.model = None
        self.outputPath = None
        # print("Solver() "+str(self.type))

    def isGurobi(self):
         return self.type == "gurobi"

    def initSolver(self, mipGap=None, timeLimit=None, outputPath=None):
        self.outputPath = outputPath
        print(f"initSolver() outputPath {self.outputPath}")
        if self.isGurobi():
            gurobiLogFile = f"{self.outputPath}gurobiLog.txt"
            if os.path.exists(gurobiLogFile):
                os.remove(gurobiLogFile)
            self.model = gp.Model()
            self.model.Params.logFile = gurobiLogFile
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

    def getStatusString(self):
        if self.isGurobi():
            # Gurobi returns an integer. We map it to a readable string.
            status_code = self.model.Status
            if status_code == GRB.OPTIMAL: return "OPTIMAL"
            if status_code == GRB.INFEASIBLE: return "INFEASIBLE"
            if status_code == GRB.UNBOUNDED: return "UNBOUNDED"
            if status_code == GRB.INF_OR_UNBD: return "INFEASIBLE_OR_UNBOUNDED"
            if status_code == GRB.TIME_LIMIT: return "TIME_LIMIT"
            if status_code == GRB.ITERATION_LIMIT: return "ITERATION_LIMIT"
            if status_code == GRB.NODE_LIMIT: return "NODE_LIMIT"
            if status_code == GRB.USER_OBJ_LIMIT: return "USER_OBJ_LIMIT"
            if status_code == GRB.INTERRUPTED: return "INTERRUPTED"
            return f"UNKNOWN_STATUS_CODE_{status_code}"
        else:
            # SCIP natively returns a readable string
            return self.model.getStatus().upper()

    def getRunStats(self):
        """Returns a dictionary containing the status, objective, gap, and time."""
        stats = {
            "status": self.getStatusString(),
            "solveTime": 0.0,
            "objective": None,
            "gap": None
        }

        if self.isGurobi():
            stats["solveTime"] = self.model.Runtime

            # Gurobi safety check: Does a solution exist?
            if self.model.SolCount > 0:
                stats["objective"] = self.model.ObjVal
                stats["gap"] = self.model.MIPGap * 100.0  # Convert to %

        else:
            # SCIP Logic
            stats["solveTime"] = self.model.getSolvingTime()

            # SCIP safety check: Does a solution exist?
            if self.model.getNSols() > 0:
                stats["objective"] = self.model.getObjVal()
                stats["gap"] = self.model.getGap() * 100.0
        stats['solveTime'] = round(stats['solveTime'],2)
        stats['objective'] = round(stats['objective'],2)
        stats['gap'] = round(stats['gap'],2)
        return stats

    # def getStatusString(self):
    #     return self.model.getStatus()
    #
    # def collectSolverResultStats(self):
    #     status = self.getStatusString()
    #     solve_time = round(self.model.getSolvingTime(),2)  # Returns time in seconds
    #
    #     result = {"status": status, "solveTime": {solve_time}}
    #
    #     if self.model.getNSols() > 0:
    #         objective = round(self.model.getObjVal(),3)
    #         mipGap = round(self.model.getGap() * 100.0, 2)  # Multiply by 100 for percentage
    #         result['objective'] = objective
    #         result['gap'] = mipGap
    #     else:
    #         result['objective'] = "No solution Found"
    #         result['gap'] = "N/A"


    def solveIt(self, plannerOutputPath):
        print("solveIt()")
        if self.isGurobi():
            self.model.optimize()
            if self.model.Status != GRB.INFEASIBLE:
                self.model.write(f"{plannerOutputPath}solution.sol")
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
            self.model.write(filename)
        else:
            self.model.writeProblem()
