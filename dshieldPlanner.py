import argparse
import ast
import copy
import json
import math
from datetime import datetime, timedelta

import numpy as np

from solver import Solver
from dshieldPlannerPreprocessor import DshieldPlannerPreprocessor
import matplotlib.pyplot as plt

import os
import pandas as pd

config = {"satList": ["CYG41884","CYG41885","CYG41886","CYG41887","CYG41888","CYG41890","CYG41891"], #"CYG41889","CYG41890","CYG41891"],
          "gsList": ["AUS", "CHI", "HI"],
          # "dataPath": "/Users/rlevinso/Applications/dshield-2026-demo-main/dshield-2026-demo/",
          "rwdThreshold": None, #0.08, 0.05, #0.1,
          "timeLimit": 8 * 3600, #12 * 3600, #3600,  # secs
          "mipGap": 0.01,    # 0.03 = 3 % mipGap, 0.018 = 1.8% gap
          "obsRate": 100/60,   # % of storage filled by each observation (storage limit = 60)
          "dnlRate": 100/1200, # % of storage freed per downlink second (1200s = 20m to empty a full buffer = 60 images / 0.05 image/second)
          "powerModel": "powerConfig.txt",
          "includeEnergyConstraints": True,
          "includeGsConstraints": True,
          "cmdSetupTime": 4,  # secs
          # "includeMvars": False,
          "maxTick": None, #12 * 3600,
          "rwdPrecision": None, #5, # number of decimal points
          "solver": "gurobi"}

class DshieldPlanner:
    def __init__ (self, dshieldDemoDataDirectory, planCreationDate):
        self.planCreationDate = planCreationDate #str(datetime.now().date()).replace("-","")
        # self.planCreationDate = "20260701"  #str(datetime.now().date()).replace("-","")
        self.planExecutionDate = None
        self.dataPath = dshieldDemoDataDirectory
        startTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.plannerDemoConfigFile = f"{self.dataPath}dshield-demo-configuration/planner_config/planner-config-{self.planCreationDate}.json" # current date
        print(f"DshieldPlanner() Plan Creation Date: {self.planCreationDate}, plannerDemoConfigFile: {self.plannerDemoConfigFile}, startTime: {startTime}")
        self.config = config
        self.solver = Solver(config["solver"])
        self.satList = config["satList"]
        self.gsList = config["gsList"]
        self.orbitsPath = None
        self.activeFireRewardsPath = None
        self.preFireRewardsPath = None
        self.plannerOutputPath = None
        self.obsRate = config["obsRate"]
        self.dnlRate = config["dnlRate"]
        self.powerModel = config["powerModel"]
        self.maxTick = config["maxTick"]
        self.includeEnergyConstraints = config["includeEnergyConstraints"]
        self.includeGsConstraints = config["includeGsConstraints"]
        self.cmdSetupTime = config["cmdSetupTime"]
        self.rwdThreshold = config["rwdThreshold"]
        self.rwdPrecision = config["rwdPrecision"]
        self.targetValues = {}
        self.activeFireTargets = {}
        self.unavailableActiveFireTargets = []
        self.allTargets = set()
        self.targetTimes = {}
        self.satChoices = {}
        self.eclipses = {}
        self.allSatCycles = {}
        self.allImages = []
        self.imageCoveringTarget = {}
        self.satPlans = {}
        self.solverStartTime = datetime.now()
        self.selectedTargets = None
        self.unselectedTargets = None
        self.selectedImages = None
        self.unselectedImages = None
        self.selectedTargetCount = None
        self.unselectedTargetCount = None
        self.selectedImageCount = None
        self.unselectedImageCount = None
        self.energyCheckpoints = {}
        self.energyDnlSecs = {}
        self.energyMin = None
        self.energyMax = None
        self.initialEnergy = None
        self.totalAvailableRewards = None
        self.initialSatStates = None
        self.finalSatStates = None

        self.downlinkConflicts = []

        self.xVars = None  # x[i] = 1 --> image i in the plan (binary)
        self.yVars = None  # y[j] = 1 --> target j is in the plan (binary)
        self.saVars = None  # sa[s,k] = storage available (%) for sat s on cycle k (0 <= a <= 100)
        self.scVars = None  # sc[s,k] = storage consumed (%) for sat s on cycle k (0 <= a <= 100)
        self.spVars = None  # sp[s,k] = storage produced (%) by s at end of cycle k (0 <= a <= 100)
        self.eaVars = None  # ea[s,k] = energy available (%) for sat s on cycle k (0 <= a <= 100)
        self.eNetVars = None  # eaNet[s,k] = net energy available (%) for sat s on cycle k (0 <= a <= 100)

        if self.includeGsConstraints:
            self.zVarIndices = None
            self.uVarIndices = None
            self.wVarIndices = None
            self.zVars = None
            # self.uVars = None
            self.wVars = None
            self.pVars = None
            self.tVars = None
            self.mVars = None

    def run(self):
        # self.findSelectedActiveFireTargets()
        # return
        # self.multiHisto()
        # return
        self.readPlannerConfigFile(self.plannerDemoConfigFile)
        self.readPreFireTargetValues()
        self.readActiveFireTargetValues()
        DshieldPlannerPreprocessor(self.dataPath, self.orbitsPath, self.plannerOutputPath, self.satList, self.gsList, self.targetValues).start()  # create choice files
        self.readInputs()
        # self.removeZeroValueChoices()
        if self.rwdThreshold and self.rwdThreshold > 0:
            self.removeLowValueChoices()
        # self.collectDownlinkWindows()
        # self.findDownlinkConflicts()
        # return

        # self.createRewardHistogram() #"./inputs/exp2_revised/results/10.28.24/exp2.optimal/")
        # self.createResidualRewardHistogram(file="./inputs/exp3/results/exp3.thresh.20pct.optimal.1min.txt")
        # return
        # dc = DataCollector(self.allImages)
        # dc.run()
        # return
        self.createDataCycles()
        self.readInitialSatStates()
        # self.printCycles()
        self.createModel()
        self.solverStartTime = datetime.now()
        timelimit = self.config["timeLimit"]
        if timelimit:
            timelimit = str(timelimit/3600) + " hrs"
        print("\n-----\nStart Time: "+self.solverStartTime.strftime('%Y-%m-%d %H:%M:%S')+", time limit: "+str(timelimit))

        # # TUNING
        # self.solver.tuneModel(18 * 3600)
        # return
        # self.createRewardHistogramNew() #"./inputs/exp2_revised/results/10.28.24/exp2.optimal/")
        # self.plotTest()
        # return
        # self.printRewardStats()
        self.solver.solveIt(f"{self.dataPath}{self.plannerOutputPath}")
        solverResult = self.extractSolution()
        self.collectSatPlans()
        self.simulatePlan()
        self.calculateLatencies()
        self.writeSatPlanFiles()
        self.printCyclesAndCollectFinalStates(solverResult)
        self.writeLatencies()
        self.writeFinalSatStates()
        # self.writeSelectedTargets()

        # self.createResidualRewardHistogram(file="exp3.threshold.0.limit.24hr.txt")
        return
        # self.showSelectedTargetRewardHistogram()
        # self.reportResults()

    def readPlannerConfigFile(self, filename):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                self.orbitsPath = data['inputs']['orbits']
                self.activeFireRewardsPath = data['inputs']['active_fire_priority']
                self.preFireRewardsPath = data['inputs']['pre_fire_priority']
                self.plannerOutputPath = data['outputs']['planner']
                self.planExecutionDate =  self.plannerOutputPath.split("/")[-2]
        else:
            print(f"readPlannerConfigFile() file not found: {filename}")

    def createModel(self):
        print("createModel()")
        outputPath = f"{self.dataPath}{self.plannerOutputPath}"
        self.solver.initSolver(config["mipGap"], config["timeLimit"], outputPath)
        self.createVariablesAndObjective()
        self.solver.setObjectiveSense("maximize")
        self.createConstraints()
        self.solver.writeModel(f"{outputPath}dshieldFire.lp")

    def createVariablesAndObjective(self):
        print("createVariablesAndObjective()")

        # xVars: x[i] = 1 --> image i is in the plan
        imageVarIndices = [i for i in range(1, self.imageCount+1)]
        self.xVars = self.solver.addBinaryVars(imageVarIndices, "x")

        # yVars: y[j] = 1 --> target j is in the plan
        targetKeys = list(self.targetTimes.keys())
        targets = [x for x in targetKeys if not isinstance(x, str)] # filter out DNL targets like HI and AUS
        sortedTargets = sorted(targets)
        missingTargets = []
        yVarObjectives = []
        for target in sortedTargets:
            if target in self.targetValues:
                if target in self.activeFireTargets:
                    rwd = self.activeFireTargets[target]
                else:
                    rwd = self.targetValues[target]
                yVarObjectives.append(rwd)
            else:
                print("ERROR! target not found in targetValues: "+str(target))
                missingTargets.append(target)
        self.yVars = self.solver.addBinaryObjectiveVars(sortedTargets, yVarObjectives, "y")

        dataVarIndices = []
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                dataVarIndices.append((s,k))

        # saVars: sa[s,k] available storage on sat s for cycle k
        self.saVars = self.solver.addContinuousVars(dataVarIndices, "sa", 0, 100)

        # scVars: sc[s,k] storage % consumed on sat s during cycle k
        self.scVars = self.solver.addContinuousVars(dataVarIndices, "sc", 0, 100)

        # spVars: sp[s,k] storage % produced on sat s at end of cycle k
        objective = [(200 * (k+1)) for s,k in dataVarIndices]
        self.spVars = self.solver.addContinuousObjectiveVars(dataVarIndices, objective, "sp", 0, 100)

        # self.spVars = self.solver.addContinuousVars(dataVarIndices, "sp", 0, 100)
        # # set objective to maximize storage produced on last cycle
        # for s in self.satList:
        #     k = self.cycleCount(s)-1
        #     spVar = self.spVars[(s,k)]
        #     spVar.Obj = 1000

        if self.includeGsConstraints:
            self.createGsVariables()

        if self.includeEnergyConstraints:
            self.createEnergyVariables(dataVarIndices)


    def createEnergyVariables(self, varIndices):

        # eaVars: ea[s,k] energy available on sat s for cycle k, after being capped at 100 %
        minChargePct = self.powerModel["minChargePct"]
        self.eaVars = self.solver.addContinuousVars(varIndices, "ea", minChargePct, 100)

        # eNet[s,k] net energy change on sat s for cycle k
        maxCycleDur = self.getMaxCycleDurationNoEclipse()
        maxEnergyProduction = maxCycleDur * self.powerModel["powerInPct"]
        ub = 100 + maxEnergyProduction
        self.eNetVars = self.solver.addContinuousVars(varIndices, "eNet", 0, ub)

        # internal vars to implement min() constraint
        # eauVars: eaRaw[s,k] <= 100   raw energy available is less than or equal to 100 %
        self.eauVars = self.solver.addBinaryVars(varIndices, "eaU")  # 1 --> energy <= 100

        # eaoVars: eaRaw[s,k] > 100    raw energy available is more than 100 %     1 --> energy > 100
        # self.eaoVars = self.solver.addBinaryVars(varIndices, "eaO")



    def createGsVariables(self):
        # zVars: z[s,k,g,n] = 1 --> sat s downlinks to g during the nth downlink window of cycle k
        self.zVarIndices = []
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                for g in self.gsList:
                    cycleK = (self.allSatCycles[s][k])
                    dnlWindowCount = len(cycleK["dnl"])
                    for n in range(dnlWindowCount):
                        dWindow = cycleK["dnl"][n]
                        if g == dWindow["gs"]:
                            self.zVarIndices.append((s, k, g, n))
        self.zVars = self.solver.addBinaryVars(self.zVarIndices, "z")

        self.vVarIndices = []  # ignores permutations
        self.wVarIndices = []  # includes permutations
        for g in self.gsList:
            for s1, k1, g1, n1 in self.zVarIndices:
                if g1 == g:
                    cycle1 = self.allSatCycles[s1][k1]
                    w1 = cycle1["dnl"][n1]
                    for s2, k2, g2, n2 in self.zVarIndices:
                        if g2 == g and s1 != s2:
                            cycle2 = self.allSatCycles[s2][k2]
                            w2 = cycle2["dnl"][n2]
                            if self.dnlWindowOverlap(w1, w2) > 0:
                                self.wVarIndices.append((g,s1,k1,n1,s2,k2,n2))
                                # ignore reverse permutation for vVars
                                if (g, s2, k2, n2, s1, k1, n1) not in self.vVarIndices:
                                    self.vVarIndices.append((g, s1, k1, n1, s2, k2, n2))
        self.vVars = self.solver.addBinaryVars(self.vVarIndices, "v")
        self.wVars = self.solver.addBinaryVars(self.wVarIndices, "w")

        # pVars: p[s,k,g,n] = planned downlink seconds for d[s,k,g,n]
        # tVars: t[s,k,g,n] = start time for downlink in d[s,k,g,n]
        self.pVars = {}
        self.tVars = {}
        for s, k, g, n in self.zVarIndices:
            slotEnd = self.dnlSlotEnd(s,k,n)
            dur = self.dnlSlotDuration(s,k,n)
            index = "[" + str(s) + "," + str(k) + "," + str(g) + "," + str(n) + "]"
            pVar = self.solver.addContinuousVar("p" + index, 0, dur)
            self.pVars[(s, k, g, n)] = pVar
            tVar = self.solver.addContinuousVar("t" + index, 0, slotEnd)
            self.tVars[(s, k, g, n)] = tVar

        # if self.includeMvars:
        #     self.mVars = {} # sum of all pVars for s,k
        #     for s in self.satList:
        #         for k in range(self.cycleCount(s)):
        #             cycle = self.allSatCycles[s][k]
        #             ub = self.getDnlTickCount(cycle)
        #             index = "["+str(s)+","+str(k)+"]"
        #             mVar = self.solver.addContinuousVar("m" + index, 0, ub)
        #             self.mVars[(s, k)] = mVar

    def createConstraints(self):
        print("createConstraints()")

        # if target j is in plan, then at least one image containing j is in the plan
        # y[j] <= sum(X[i] for all images i containing target j)                        (1)
        for target in self.yVars:
            images = self.imagesContainingTarget(target)
            self.solver.addConstraint(self.yVars[target] <= sum(self.xVars[i] for i in images), "c1.targetImageInPlan." +str(target))

        # Storage used on sat s on cycle k = sum of storage used by s on cycle k
        # sc[s,k] = sum( obsRate * x[i] for i in sat s cycle k                          (2)
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = self.allSatCycles[s][k]
                firstImage, lastImage = self.firstAndLastImagesInCycle(cycle)
                self.solver.addConstraint(self.scVars[s,k] == sum(self.obsRate * self.xVars[i] for i in range(firstImage, lastImage + 1)),
                                              "c2.storageUsedOnCycle." + s +"." + str(k+1))

        # sa[s,0] = 100 (default, or from self.initialSatStates)      (3)
        for s in self.satList:
            if self.initialSatStates:
                initialStorageAvailable = self.initialSatStates[s]['storage']
            else:
                initialStorageAvailable = 100
            self.solver.addConstraint(self.saVars[s,0] == initialStorageAvailable, "c3.availableStorageFirstCycle."+s)

        # Available storage for cycle k = available storage for prior cycle - storage used on prior cycle + freed storage at end of prior cycle
        # sa[s,k] = sa[s,k-1] - sc[s,k-1] + sp[s,k-1]                                    (4)
        for s in self.satList:
            for k in range(1,self.cycleCount(s)):
                self.solver.addConstraint(self.saVars[s,k] == self.saVars[s,k-1] - self.scVars[s,k-1] + self.spVars[s,k-1], "c4.availStorageCycleStart."+s+"."+str(k))

        # used storage must never exceed available storage for any cycle
        # sc[s,k] <= sa[s,k]                                                         (5)
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                self.solver.addConstraint(self.scVars[s, k] <= self.saVars[s, k], "c5.neverExceedAvailStorage." + s + "." + str(k))

        # cannot downlink when storage is empty
        # sp[s,k] <= 100 - (sa[s,k] - sc[s,k])                                        (6)
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                self.solver.addConstraint(self.spVars[s, k] <= 100 - (self.saVars[s, k] - self.scVars[s,k]), "c6.cannotDownlinkIfStorageEmpty." + s + "." + str(k))

        # if self.includeMvars:
        #     for s in self.satList:
        #         for k in range(self.cycleCount(s)):
        #             cycle = self.allSatCycles[s][k]
        #             vars = []
        #             if len(cycle["dnl"]) > 0:
        #                 for n in range(len(cycle["dnl"])):
        #                     g = cycle["dnl"][n]["gs"]
        #                     vars.append(self.pVars[(s, k, g, n)])
        #             self.solver.addConstraint(self.mVars[s, k] == sum(vars), "c30.mVar=sum(pVars)." + s + "." + str(k))

        # cannot downlink longer than downlink duration
        if self.includeGsConstraints:
            self.createGsConstraints()
        else:
            # sp[s,k] <= dnlRate * dnlTicks(s,k)                                        (7)
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = self.allSatCycles[s][k]
                    dnlTicks = self.getDnlTickCount(cycle) # len(cycle["dnl"])  if "dnl" in cycle else 0
                    self.solver.addConstraint(self.spVars[s, k] <= self.dnlRate * dnlTicks, "c7.cannotExceedDownlinkDuration." + s + "." + str(k))

        # ENERGY CONSTRAINTS
        if self.includeEnergyConstraints:
            self.createEnergyConstraints()

        # Command Separation Constraints
        # if self.includeSensorModes:
        self.createCommandMutexConstraints()

    def createGsConstraints(self):
        # sp[s,k] == dnlRate * sum[pVar in pVars[s,*,k,*]                                (7)
        # if self.includeMvars:
        #     for s in self.satList:
        #         for k in range(self.cycleCount(s)):
        #             mVar = self.mVars[s,k]
        #             self.solver.addConstraint(self.spVars[s, k] == self.dnlRate * mVar,
        #                                       "c7m.cannotExceedDownlinkDuration." + s + "." + str(k))
        # else:
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = self.allSatCycles[s][k]
                vars = []
                if len(cycle["dnl"]) > 0:
                    for n in range(len(cycle["dnl"])):
                        g = cycle["dnl"][n]["gs"]
                        vars.append(self.pVars[(s, k, g, n)])
                self.solver.addConstraint(self.spVars[s, k] == self.dnlRate * sum(vars),
                                          "c7.cannotExceedDownlinkDuration." + s + "." + str(k))

        # z[s,k,g,n] = 1 --> p[s,k,g,n] > 0:  z[p,s,k,n] <= p[s,k,g,n]
        # for index in self.zVarIndices:
        #     s,k,g,n = index
        #     M = self.dnlSlotDuration(s,k,n)+1
        #     indexName = "["+str(s)+","+str(k)+","+str(g)+","+str(n)+"]"
        #     self.solver.addConstraint(self.zVars[index] <= self.pVars[index], "c27.p"+indexName+"=0->z"+indexName+"=0")

        #z[s,k,g,n] = 0 --> p[s,g,k,n] = 0:   p[s,g,k,n] <= M * z[s,k,g,n]
        for index in self.zVarIndices:
            s,k,g,n = index
            M = self.dnlSlotDuration(s,k,n)+10
            indexName = "["+str(s)+","+str(k)+","+str(g)+","+str(n)+"]"
            self.solver.addConstraint(self.pVars[index] <= M * self.zVars[index], "c20.z"+indexName+"=0->p"+indexName+"=0")

            self.solver.addConstraint(self.zVars[index] <= self.pVars[index], "c21.p"+indexName+"-=0->z"+indexName+"=0")

        # z[s,k,g,n] = 0 --> t[s,g,k,n] = 0:   t[s,g,k,n] <= M * z[s,k,g,n]
        for index in self.zVarIndices:
            s,k,g,n = index
            M = self.dnlSlotEnd(s,k,n)+1
            indexName = "["+str(s)+","+str(k)+","+str(g)+","+str(n)+"]"
            self.solver.addConstraint(self.tVars[index] <= M * self.zVars[index], "c21.z"+indexName+"=0->t"+indexName+"=0")

        # z[s,k,g,n] = 1 --> d[s,k,g,n].start <= t[s,g,k,n] : d[s,k,g,n].start - t[s,k,g,n] <= M (1-z[s,k,g,n])
        for index in self.zVarIndices:
            s,k,g,n = index
            slotStart = self.dnlSlotStart(s,k,n)
            M = slotStart+1
            indexName = "["+str(s)+","+str(k)+","+str(g)+","+str(n)+"]"
            self.solver.addConstraint(slotStart - self.tVars[index] <= M * (1-self.zVars[index]), "c22.z"+indexName+"=1->d"+indexName+".start<t"+indexName)

        # z[s,k,g,n] = 1 --> t[s,k,g,n] + p[s,k,g,n] <= d[s,k,g,n].end : t[s,g,k,n] + p[s,k,g,n] - d[s,k,g,n].end -  <= M (1-z[s,k,g,n])
        for index in self.zVarIndices:
            s,k,g,n = index
            indexName = "["+str(s)+","+str(k)+","+str(g)+","+str(n)+"]"
            slotEnd = self.dnlSlotEnd(s,k,n)
            M = slotEnd+1
            self.solver.addConstraint(self.tVars[index] + self.pVars[index] - slotEnd <= M * (1-self.zVars[index]), "c23.z"+indexName+"=1->t"+indexName+"+d"+indexName+"<d"+indexName+".end")

        # u[g,s1,k1,n1,s2,k2,n2] <= z[s1,k1,g,n1]                       (24a)
        # u[g,s1,k1,n1,s2,k2,n2] <= z[s2,k2,g,n2]                       (24b)
        # u[g,s1,k1,n1,s2,k2,n2] >= z[s1,k1,g,n1] + z[s2,k2,g,n2] - 1   (24c)
        # w[g,s1,k1,n1,s2,k2,n2] + w[g,s2,k2,n2,s1,k1,n1] -1 <= M (1 - u[g,s1,k1,n1,s2,k2,n2]) (25a)
        # w[g,s1,k1,n1,s2,k2,n2] + w[g,s2,k2,n2,s1,k1,n1] -1 >= M (1 - u[g,s1,k1,n1,s2,k2,n2]) (25b)
        for index in self.vVarIndices:
            vVar = self.vVars[index]
            g, s1, k1, n1, s2, k2, n2 = index
            zVar1 = self.zVars[(s1,k1,g,n1)]
            zVar2 = self.zVars[(s2,k2,g,n2)]
            vVarName = "v[" + str(g) + "," + str(s1) + "," + str(k1) + "," + str(n1) + "," + str(
                s2) + "," + str(k2) + "," + str(n2) + "]"
            self.solver.addConstraint(vVar <= zVar1, "c14a."+vVarName)
            self.solver.addConstraint(vVar <= zVar2, "c14b."+vVarName)
            self.solver.addConstraint(zVar1 + zVar2 - 1 <= vVar, "c14c."+vVarName)

            wVar1 = self.wVars[(g,s1,k1,n1,s2,k2,n2)]
            wVar2 = self.wVars[(g,s2,k2,n2,s1,k1,n1)]
            self.solver.addConstraint(wVar1 + wVar2 == vVar, "c15."+vVarName)

            # M = 4
            # wVar1 = self.wVars[(g,s1,k1,n1,s2,k2,n2)]
            # wVar2 = self.wVars[(g,s2,k2,n2,s1,k1,n1)]
            # wVarName = "w[" + str(g) + "," + str(s1) + "," + str(k1) + "," + str(n1) + "," + str(
            #     s2) + "," + str(k2) + "," + str(n2) + "]"
            # self.solver.addConstraint(wVar1 + wVar2 - 1 <= M * (2 - zVar1 - zVar2), "c25a."+wVarName)
            # self.solver.addConstraint(wVar1 + wVar2 - 1 >= M * (2 - zVar1 - zVar2), "c25b."+wVarName)
            # # self.solver.addConstraint(wVar1 + wVar2 - 1 >= M * (1 - uVar), "c25b."+uVarName)

            # self.solver.addConstraint(
            #         self.uVars[g, s1, k1, n1, s2, k2, n2] + self.uVars[g, s2, k2, n2, s1, k1, n1] == 1, consName)

        # t[s1,k1,g,n1] + p[s1,k1,g,n1] + setup[g] - t[s2,k2,g,n2] <= M (1- w[g,s1,k1,n1,s2,k2,n2]), where s1 != s2
        for index in self.wVarIndices:
            g,s1,k1,n1,s2,k2,n2 = index
            setup = 5
            M = self.dnlSlotEnd(s1,k1,n1)+setup+10
            t1VarIndexName = "["+str(s1)+","+str(k1)+","+str(g)+","+str(n1)+"]"
            t2VarIndexName = "["+str(s2)+","+str(k2)+","+str(g)+","+str(n2)+"]"
            wVarIndexName = "["+str(g)+","+str(s1)+","+str(k1)+","+str(n1)+","+str(s2)+","+str(k2)+","+str(n2)+"]"
            consName = "c16.w"+wVarIndexName+"=1->t"+t1VarIndexName+"+p"+t1VarIndexName+"+setup<=t"+t2VarIndexName
            self.solver.addConstraint(self.tVars[s1,k1,g,n1] + self.pVars[s1,k1,g,n1] + setup - self.tVars[s2,k2,g,n2] <= M * (1 - self.wVars[index]), consName)

        # # u[g,s1,k1,n1,s2,k2,n2] + u[g,s2,k2,n2,s1,k1,n1] = 1
        # coveredIndices = []
        # # skippedIndices = []
        # for index in self.uVarIndices:
        #     g,s1,k1,n1,s2,k2,n2 = index
        #     if (s1,k1,n1) not in coveredIndices:
        #         coveredIndices.append((s2,k2,n2))
        #         u1VarIndexName = "["+str(g)+","+str(s1)+","+str(k1)+","+str(n1)+","+str(s2)+","+str(k2)+","+str(n2)+"]"
        #         u2VarIndexName = "["+str(g)+","+str(s2)+","+str(k2)+","+str(n2)+","+str(s1)+","+str(k1)+","+str(n1)+"]"
        #         consName = "c25.u"+u1VarIndexName+"+"+u2VarIndexName+"=1"
        #         self.solver.addConstraint(self.uVars[g,s1,k1,n1,s2,k2,n2] + self.uVars[g,s2,k2,n2,s1,k1,n1]  ==1, consName)
        #     else:
        #         skippedIndices.append((s1,k1,n1))
        # print("hey")

    def createCommandMutexConstraints(self):
        # ensure minSeparation of overlapping observations
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = self.allSatCycles[s][k]
                firstImage, lastImage = self.firstAndLastImagesInCycle(cycle)
                overlappingImages = []
                for imageId1 in  range(firstImage, lastImage+1):
                    image1 = self.allImages[imageId1 - 1]
                    for imageId2 in range(imageId1+1, lastImage+1):
                        image2 = self.allImages[imageId2 - 1]
                        if image2["time"] - image1["time"] < self.cmdSetupTime and image1["type"] != image2["type"]:
                            overlappingImages.append((imageId1, imageId2))
                for imageId1, imageId2 in overlappingImages:
                    assert imageId1 < imageId2, "createCommandMutexConstraints() ERROR: duplicat image ids:"+str(imageId1)
                    x1 = self.xVars[imageId1]
                    x2 = self.xVars[imageId2]
                    self.solver.addConstraint(x1 + x2 <= 1, "cx.cmdMutex." + s+"."+str(k+1)+"."+str(imageId1)+"."+str(imageId2))

    def createEnergyConstraints(self):
        # ea[s,0] = 100      Available energy is 100% for first cycle on all sats   (8)
        for s in self.satList:
            if self.initialSatStates:
                initialEnergyAvailable = self.initialSatStates[s]['energy']
            else:
                initialEnergyAvailable = 100
            self.solver.addConstraint(self.eaVars[s, 0] == initialEnergyAvailable, "c8.availableEnergyFirstCycle." + s)

        # # never dip below miniumum energy threshold
        # # ea[s,k] >= minimum energy threshold                                       (9)
        # minChargePct = self.powerModel["minChargePct"]
        # for s in self.satList:
        #     for k in range(self.cycleCount(s)):
        #         self.solver.addConstraint(self.eaVars[s, k] >= minChargePct,
        #                                   "c9.energyMinThreshold." + s + "." + str(k))

        # eNet[s,k] = energy produced in cycle k - energy consumed in cycle k         (10)
        powerInPct = self.powerModel["powerInPct"]
        powerOutPct = self.powerModel["powerOutPct"]
        powerOutDnl = self.powerModel['powerOutDnlPct']
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = self.allSatCycles[s][k]
                cycleDur = self.getCycleDuration(cycle)
                cycleEclipseDur = self.getEclipseTickCount(s, cycle["start"], cycle["end"])
                powerIn = (cycleDur - cycleEclipseDur) * powerInPct
                powerOutDefault = cycleDur * powerOutPct
                cycle["powerIn"] = powerIn  # stash for later printing
                cycle["powerOutDefault"] = powerOutDefault  # stash for later printing
                cycle["eclipseDur"] = self.getEclipseTickCount(s, cycle["start"], cycle["end"])
                # if self.includeMvars:
                #     mVar = self.mVars[s, k]
                #     self.solver.addConstraint(self.eNetVars[s, k] == self.eaVars[s,k] + powerIn - powerOutDefault
                #                               - (powerOutDnl * mVar), "c10m.netEnergy." + s + "." + str(k))
                # else:
                pList = []
                for n in range(len(cycle["dnl"])):
                    g = cycle["dnl"][n]["gs"]
                    pList.append(self.pVars[(s, k, g, n)])
                self.solver.addConstraint(self.eNetVars[s, k] == self.eaVars[s, k] + powerIn - powerOutDefault
                                          - (powerOutDnl * sum(pList)), "c10.netEnergy." + s + "." + str(k))

        # # TODO: BUG? constraint uses k vs. k-1? No, maybe, because the k-1 happens next constraint
        # # eaRaw[s,k] = ea[s,k-1] + eNet[s,k-1] Available energy for cycle k = available energy for prior cycle + energy produced prior cycle - energy consumed prior cycle        (10)
        # for s in self.satList:
        #     for k in range(self.cycleCount(s)):
        #         self.solver.addConstraint(self.eaRawVars[s, k] == self.eaVars[s, k] + self.eNetVars[s, k],
        #                                   "c.11.eRaw." + s + "." + str(k))

        # ea[s,k] = min(eNet[s,k-1], 100)  energy is capped at 100 % despite constant solar panel exposure    (12)
        self.createMinValueConstraints()

        self.createActiveFireConstraints()

    def createActiveFireConstraints(self):
        for target in self.activeFireTargets:
            if target in self.yVars:
                yVar = self.yVars[target]
                self.solver.addConstraint(yVar == 1, "c.30.activeFire."+str(target))
            else:
                self.unavailableActiveFireTargets.append(target)

    def createMinValueConstraints(self):
        # ea[s,k] = min(eNet[s,k-1], 100)  energy is capped at 100 % despite constant solar panel exposure    (12)
        # from https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
        for s in self.satList:
            for k in range(1, self.cycleCount(s)):
                index = "[" + str(s) + "," + str(k) + "]"
                M = 1000
                # bind eau indicator = 1 <--> 100 <= eNet[s,k-1]
                self.solver.addConstraint(100 - self.eNetVars[s, k - 1] <= M * (1 - self.eauVars[s, k]),
                                          "c.12a.energyCap" + index)
                self.solver.addConstraint(self.eNetVars[s, k - 1] - 100 <= M * self.eauVars[s, k],
                                          "c.12b.energyCap" + index)

                # self.solver.addConstraint(self.eaVars[s,k] <= 100, "c.12.energyCap") # redundant with var UB = 100
                self.solver.addConstraint(self.eaVars[s, k] <= self.eNetVars[s, k - 1], "c.12c.energyCap")
                self.solver.addConstraint(self.eaVars[s, k] >= 100 - M * (1 - self.eauVars[s,k]), "c.12d.energyCap")
                self.solver.addConstraint(self.eaVars[s, k] >= self.eNetVars[s,k-1] - M *  self.eauVars[s,k], "c.12e.energyCap")

    # def createMinValueConstraintsOld(self):
    #     # ea[s,k] = min(eNet[s,k-1], 100)  energy is capped at 100 % despite constant solar panel exposure    (12)
    #     for s in self.satList:
    #         for k in range(1, self.cycleCount(s)):
    #             self.solver.addConstraint(self.eaVars[s, k] <= self.eNetVars[s, k - 1], "c.12a.energyCap")
    #             # TODO: is this redundant with var bounds?
    #             # self.solver.addConstraint(self.eaVars[s,k] <= 100, "c.12b.energyCap")
    #             self.solver.addConstraint(
    #                 self.eaVars[s, k] >= self.eNetVars[s, k - 1] - 1000 * (1 - self.eauVars[s, k]), "c.12c.energyCap")
    #             self.solver.addConstraint(self.eaVars[s, k] >= 100 - 100 * (1 - self.eaoVars[s, k]), "c.12d.energyCap")
    #             self.solver.addConstraint(self.eauVars[s, k] + self.eaoVars[s, k] == 1, "c.12e.energyCap")

    def dnlSlotDuration(self, s,k,n):
        cycle = self.allSatCycles[s][k]
        dnlWindow = cycle["dnl"][n]
        dur = dnlWindow["end"] - dnlWindow["start"] + 1
        return dur

    def dnlSlotEnd(self, s,k,n):
        cycle = self.allSatCycles[s][k]
        dnlWindow = cycle["dnl"][n]
        return dnlWindow["end"]

    def dnlSlotStart(self, s,k,n):
        cycle = self.allSatCycles[s][k]
        dnlWindow = cycle["dnl"][n]
        return dnlWindow["start"]

    def getDnlTickCount(self, cycle):
        # called by createConstraints() and printCycles()
        tickCount = 0
        if "dnl" in cycle:
            for dnlWindow in cycle["dnl"]:
                dur = dnlWindow["end"]-dnlWindow["start"]+1
                tickCount += dur
        return tickCount

    def testWindowOverlap(self, w1, w2):
        w1start = w1[0]
        w1end = w1[1]
        w2start = w2[0]
        w2end = w2[1]
        return max(0, min(w1end+1, w2end+1) - max(w1start, w2start))

    def dnlWindowOverlap(self, w1, w2):
        w1start = w1["start"]
        w1end = w1["end"]
        w2start = w2["start"]
        w2end = w2["end"]
        return max(0, min(w1end+1, w2end+1) - max(w1start, w2start))

    def getCycleEndTime(self, cycle):
        # called only by addCycleStartAndEndTimes()
        if cycle["dnl"]:
            lastDnlWindow = cycle["dnl"][-1]
            return lastDnlWindow["end"]
        else:
            lastObservation = cycle["obs"][-1]
            return lastObservation["time"]

    def getCycleEndTimeOld(self, cycle):
        # called only by addCycleStartAndEndTimesOld()
        if cycle["dnl"]:
            return cycle["dnl"][-1]
        else:
            lastObservation = cycle["obs"][-1]
            return lastObservation["time"]

    def getEclipseTickCount(self, sat, start, end):
        satEclipses = self.eclipses[sat]
        count = 0
        for time in satEclipses:
            if time < start:
                continue
            elif time > end:
                return count
            else:
                count += 1
        return count

    def extractSolution(self):
        print("extractSolution()")
        self.selectedTargets = self.solver.extractSelectedBinaryVars(self.yVars)
        self.unselectedTargets = self.solver.extractUnselectedBinaryVars(self.yVars)
        self.selectedImages = self.solver.extractSelectedBinaryVars(self.xVars)
        self.unselectedImages= self.solver.extractUnselectedBinaryVars(self.xVars)
        self.selectedTargetCount = len(self.selectedTargets)
        self.unselectedTargetCount = len(self.unselectedTargets)
        self.selectedImageCount = len(self.selectedImages)
        self.unselectedImageCount = len(self.unselectedImages)
        print("\nTargets: "+str(self.selectedTargetCount)+" selected + " +str(self.unselectedTargetCount) +" unselected = " + str(self.selectedTargetCount + self.unselectedTargetCount))
        print("Images: "+str(self.selectedImageCount)+" selected + "+str(self.unselectedImageCount) +" unselected = " + str(self.selectedImageCount + self.unselectedImageCount))
        self.updateCyclesWithPlan()
        self.validateSolution(self.selectedTargets, self.selectedImages)
        result = self.solver.getRunStats()
        return result

    def collectSatPlan(self, sat):
        plan = []
        for image in self.satPlans[sat]:
            cmdTime = image['time']
            plan.append((cmdTime, "RawIF"))
        for k in range(self.cycleCount(sat)):
            satCycle = self.allSatCycles[sat][k]
            dnlPlan = satCycle["dnlPlan"] if "dnlPlan" in satCycle else None
            if dnlPlan:
                dnlCmdCount = len(dnlPlan['z'])
                for i in range(dnlCmdCount):
                    t = dnlPlan['t'][i]
                    p = dnlPlan['p'][i]
                    target = t[0][2]
                    dnlStart = int(t[1])
                    dnlEnd = int(dnlStart + p[1])
                    for t in range(dnlStart, dnlEnd):
                        plan.append((t, f"DNL: {target}"))
        plan.sort()
        return plan

    def writeSatPlanFiles(self):
        outputPath = self.plannerOutputPath
        for sat in self.satList:
            satPlan = self.collectSatPlan(sat)
            filepath = f"{self.dataPath}{outputPath}"
            filename = f"{filepath}{sat}_plan.csv"
            with open(filename, "w") as f:
                f.write(f"# {sat} plan for {self.planExecutionDate}\n# Time, Command\n")
                for time, cmd in satPlan:
                    f.write(f"{time}, {cmd}\n")

    def writeSatPlanFilesOld(self):
        outputPath = self.plannerOutputPath
        for sat in self.satList:
            filepath = f"{self.dataPath}{outputPath}"
            filename = f"{filepath}{sat}_plan.csv"
            with open(filename, "w") as f:
                f.write(f"# {sat} plan for {self.planExecutionDate}\n# Time, Command\n")
                for k in range(self.cycleCount(sat)):
                    satCycle = self.allSatCycles[sat][k]
                    for image in satCycle['obs']:
                        time = image['time']
                        f.write(f"{time}, RawIF\n")
                    dnlPlan = satCycle["dnlPlan"] if "dnlPlan" in satCycle else None
                    if dnlPlan:
                        dnlCmds = []
                        for cmd, startTime in dnlPlan['t']:
                            dnlCmds.append({"startTime": int(startTime), "target": cmd[2]})
                        for i, p in enumerate(dnlPlan['p']):
                            dur = int(p[1])
                            dnlCmds[i]['dur'] = dur
                            dnlCmds[i]['endTime'] = int(dnlCmds[i]['startTime'] + dur)

                        for dnlCmd in dnlCmds:
                            startTime = dnlCmd['startTime']
                            endTime = dnlCmd['endTime']
                            target = dnlCmd['target']
                            msg = f"{startTime}-{endTime}, DNL_{target}\n"
                            f.write(msg)


    def writeFinalSatStates(self):
        filename = f"{self.dataPath}{self.plannerOutputPath}finalSatStates.json"
        with open(filename, "w") as f:
            json.dump(self.finalSatStates, f, indent=4)
        print("\nFinal Sat States:")
        for sat in self.finalSatStates:
            state = self.finalSatStates[sat]
            print(f"  {sat} storage: {state['storage']}, energy: {state['energy']}")

    def readInitialSatStates(self):
        date = datetime.strptime(self.planExecutionDate, "%Y%m%d")
        previousDate = date - timedelta(days=1)
        priorDate = previousDate.strftime("%Y%m%d")
        priorDateFolder = f"{self.dataPath}/planner/output/{priorDate}/"
        if os.path.exists(priorDateFolder):
            filename = priorDateFolder + "finalSatStates.json"
            if os.path.exists(filename):
                print(f"\nreadInitialSatStates() file: {filename}")
                with open(filename, "r") as file:
                    self.initialSatStates = json.load(file)
                print("Initial Sat States:")
                for sat in self.initialSatStates:
                    state = self.initialSatStates[sat]
                    print(f"  {sat} storage: {state['storage']}, energy: {state['energy']}")
            else:
                print(f"\nreadInitialSatStates() prior state file not found: {filename} ")
        else:
            self.initialSatStates = {"CYG41884": {"storage": 100.0, "energy": 100},
                                     "CYG41885": {"storage": 100.0, "energy": 100},
                                     "CYG41886": {"storage": 100.0, "energy": 100},
                                     "CYG41887": {"storage": 100.0, "energy": 100},
                                     "CYG41888": {"storage": 100.0, "energy": 100},
                                     "CYG41890": {"storage": 100.0, "energy": 100},
                                     "CYG41891": {"storage": 100.0, "energy": 100}}

            print(f"\nreadInitialSatStates() prior date folder not found: {priorDateFolder} ")

    def updateCyclesWithPlan(self):
        for i in self.selectedImages:
            image = self.getImage(i)
            sat = image["sat"]
            k = image["cycle"]
            satCycle = self.allSatCycles[sat][k]
            if "selectedImages" not in satCycle:
                satCycle["selectedImages"] = []
            satCycle["selectedImages"].append(i)
        for sat in self.satList:
            for k in range(self.cycleCount(sat)):
                satCycle = self.allSatCycles[sat][k]
                if "availSpace" not in satCycle:
                    satCycle["availSpace"] = None
                if "usedSpace" not in satCycle:
                    satCycle["usedSpace"] = None
                if "freedSpace" not in satCycle:
                    satCycle["freedSpace"] = None
                satCycle["availSpace"] = self.solver.getVarValue(self.saVars[sat,k])
                satCycle["usedSpace"] = self.solver.getVarValue(self.scVars[sat,k])
                satCycle["freedSpace"] = self.solver.getVarValue(self.spVars[sat,k])
                satCycle["dnlPlanSecs"] = round(satCycle["freedSpace"]/self.dnlRate,3)

                if self.includeGsConstraints:
                    satCycle["dnlPlan"] = self.collectCycleDnlPlan(sat,k)
                if self.includeEnergyConstraints:
                    satCycle["energyAvail"] = self.solver.getVarValue(self.eaVars[sat,k])
                    # satCycle["energyRaw"] = self.solver.getVarValue(self.eaRawVars[sat,k])
                    satCycle["energyNet"] = self.solver.getVarValue(self.eNetVars[sat,k])

    def collectCycleDnlPlan(self, sat, cycle):
        plan = {"z":[],"t":[],"p":[],"tNoP":[], "u":[], "w":[], "v": []}
        for index in self.zVarIndices:
            s,k,g,n = index
            if s == sat and cycle == k:
                zVal = self.solver.getVarValue(self.zVars[index])
                if zVal > 0.0:
                    plan["z"].append(index)
                tVal = self.solver.getVarValue(self.tVars[index])
                if tVal > 0.0:
                    plan["t"].append((index, tVal))
                pVal = self.solver.getVarValue(self.pVars[index])
                if pVal > 0.0:
                    plan["p"].append((index, pVal))
                else:
                    plan["tNoP"].append((index,pVal))
        return plan

    def validateSolution(self, selectedTargets, selectedImages):
        print("\nValidating solution ...")
        status = self.ensureAllSelectedTargetsCoveredBySelectedImages(selectedTargets, selectedImages)
        print("   All selected targets are covered by a selected image: "+str(status))
        status = self.ensureAllSelectedImagesCoverAllSelectedImages(selectedTargets, selectedImages)
        print("   All selected images cover all selected targets: "+str(status))
        status = self.ensureNoOverlappingCmds(selectedImages)
        print("   All no overlapping commands: "+str(status))
        # TODO: validate energy consumption


    def ensureAllSelectedTargetsCoveredBySelectedImages(self, selectedTargets, selectedImages):
        status = True
        for target in selectedTargets:
            imagesCoveringTarget = self.imagesContainingTarget(target)
            for image in imagesCoveringTarget:
                if image in selectedImages:
                    if target not in self.imageCoveringTarget:
                        self.imageCoveringTarget[target] = []
                    self.imageCoveringTarget[target].append(target)

        for target in selectedTargets:
            if target not in self.imageCoveringTarget:
                print("\n**** ensureAllSelectedTargetsCoveredBySelectedImages() ERROR! target "+str(target) + " not covered by image!")
                status = False
        return status

    def ensureAllSelectedImagesCoverAllSelectedImages(self, selectedTargets, selectedImages):
        status = True
        allCoveredTargets = set()
        for image in selectedImages:
            imageDict = self.allImages[image-1]
            for t in imageDict["targets"]:
                allCoveredTargets.add(t)

        for target in selectedTargets:
            if target not in allCoveredTargets:
                print("\n**** ensureAllSelectedImagesCoverAllSelectedImages() ERROR! target "+str(target) + " not covered by image!")
                status = False
        return status

    def ensureNoOverlappingCmds(self,selectedImages):
        overlappingImages = []
        for imageId1 in selectedImages:
            image1 = self.allImages[imageId1-1]
            sat1 = image1["sat"]
            for imageId2 in selectedImages:
                if imageId2 > imageId1:
                    image2 = self.allImages[imageId2-1]
                    sat2 = image2["sat"]
                    if sat1 == sat2:
                        if image2["time"] - image1["time"] < self.cmdSetupTime and image1["type"] != image2["type"]:
                            overlappingImages.append((image1, image2))
        if overlappingImages:
            print("ensureNoOverlappingImages() ERROR! Overlapping Images: ("+str(len(overlappingImages))+"):")
            for x in overlappingImages:
                print(str(x))
            return False
        else:
            return True


    def collectSatPlans(self):
        for imageId in self.selectedImages:
            image = self.allImages[imageId-1]
            sat = image["sat"]
            if sat not in self.satPlans:
                self.satPlans[sat] = []
            self.satPlans[sat].append(image)


    def simulatePlan(self):
        # init sat states
        self.plans = {}
        states = {}
        for sat in self.satList:
            self.plans[sat] = []
            states[sat] = {"sat": sat, "time": 0, "availSpace": 100}

        # simulate plans
        for satId in self.satPlans:
            satPlan = self.satPlans[satId]
            satState = copy.deepcopy(states[satId])
            for image in satPlan:
                # print(str(image))
                satState = copy.deepcopy(satState)
                satState["time"] = image["time"]
                obsDataRate = self.obsRate #if image["type"] == "large" else self.obsRateSmall
                satState["availSpace"] -= obsDataRate
                # if satState["availSpace"] <= 0:
                #     print("simulatePlan() ERROR! storage below empty! sat: "+satId+", image: "+str(image)+", satState: "+str(satState))
                # self.plans[satId].append({"image": image, "state": satState})

            # TODO: simulate downlinks!

    def calculateLatencies(self):
        print("\nCalculating Latencies")
        for sat in self.allSatCycles:
            self.calculateSatLatencies(sat)

    def calculateSatLatencies(self, sat):
        print("calculateSatLatencies() sat: "+sat)
        buffer = []
        cycleId = -1
        for cycle in self.allSatCycles[sat]:
            cycleId +=1
            if "selectedImages" in cycle:
                for imageId in cycle["selectedImages"]:
                    time = self.getImage(imageId)["time"]
                    buffer.append(
                        {"id": imageId, "collectionTime": time, "remainingPct": 100})
            if buffer:
                dnlPlan = cycle["dnlPlan"] if "dnlPlan" in cycle else None
                if dnlPlan:
                    dnlPlanTimes = dnlPlan["t"]
                    dnlPlanDurs = dnlPlan["p"]
                    currentImage = buffer[0]
                    print("Current image: sat " + str(sat) + ", cycle " + str(cycleId) + ", " + str(
                        currentImage)+", imageCount: "+str(len(buffer)))
                    for dnlWindow in range(len(dnlPlanTimes)):
                        if buffer and currentImage:
                            dnlStartTime = int(dnlPlanTimes[dnlWindow][1])
                            dur = dnlPlanDurs[dnlWindow][1]
                            dnlEndTime = int(dnlStartTime + dur)
                            for tick in range(dnlStartTime, dnlEndTime):
                                # subtract 5 % image remaining per second (20 seconds to downlink an image)
                                # subtract 5 % image remaining per second (20 seconds to downlink an image)
                                currentImage["remainingPct"] = round(currentImage["remainingPct"] - 5, 3)
                                if currentImage["remainingPct"] <= 0:
                                    currentImage["downlinkTime"] = tick
                                    currentImage["latency"] = round((currentImage["downlinkTime"] - currentImage["collectionTime"] + 1)/60, 2)
                                    if "latencies" not in cycle:
                                        cycle["latencies"] = []
                                    downlinkedImage = buffer.pop(0)
                                    cycle["latencies"].append(downlinkedImage)
                                    print("Downlinked image: sat "+str(sat)+", cycle "+str(cycleId)+", " + str(downlinkedImage)+", remaining image count: "+str(len(buffer)))
                                    if buffer:
                                        currentImage = buffer[0]
                                        print("Current image: sat " + str(sat) + ", cycle " + str(cycleId) + ", " + str(
                                            currentImage))
                                    else:
                                        currentImage = None
                                        print("Buffer Empty! sat " + str(sat) + ", cycle " + str(cycleId))
                                        break



    def getImage(self, imageId):
        return self.allImages[imageId-1]

    def cycleCount(self, sat):
        return len(self.allSatCycles[sat])


    #####  Read Input Data ######

    def readInputs(self):
        print("readInputs()")
        print("config: "+str(self.config))
        self.initPowerModel()
        # self.readPreFireTargetValues()
        # self.readActiveFireTargetValues()
        # self.increaseTargetValues()
        for sat in self.satList:
            self.readSatChoiceFile(sat)
            choices = list(self.satChoices[sat].keys())
            print("tp count for " + sat + ": " + str(len(choices)) + ", TP range: " + str(choices[0]) + " - " + str(
                choices[-1]))
            self.readEclipseFileForSat(sat)
        print(f"Target value count: {len(self.targetValues.keys())} (including unavailable targets)" )
        print(f"All targets count: {len(self.allTargets)}")

    def readSatChoiceFile(self, sat):
        satChoices = {}  # {TP: {sourceID: [gpList]}}
        filepath = f"{self.dataPath}{self.plannerOutputPath}/"
        if not os.path.exists(filepath):
            print("readSatChoiceFile() creating dir: "+filepath)
            os.mkdir(filepath)
        filename = filepath +sat+"_choices.txt"

        print("readSatChoiceFile() reading file for " + sat + ": " + filepath)
        with open(filename, "r") as f:
            for line in f:
                filteredLine = line.strip()
                if filteredLine and not filteredLine.startswith("--- GAP"):
                    dict = "{" + filteredLine + "}"
                    choices = ast.literal_eval(dict)
                    tp = list(choices.keys())[0]
                    if self.maxTick and tp > self.maxTick:
                        break
                    satChoices.update(choices)
        for tp in satChoices:
            for cmd in satChoices[tp]:
                if cmd['cmd'] == 'obs':
                    targets = cmd['targets']
                    self.allTargets.update(targets)
        self.satChoices[sat] = satChoices

    def readPreFireTargetValues(self):
        preFirePath = self.preFireRewardsPath
        filepath =  f"{self.dataPath}{preFirePath}TV_PRE_FIRE.csv"
        assert os.path.exists(filepath), f"\nreadPreFireTargetValues() ERROR! file not found {filepath}"
        with open(filepath, "r") as f:
            firstLine = True
            for line in f:
                if firstLine:
                    firstLine = False
                    continue
                filteredLine = line.strip()
                if filteredLine:
                    target, value = filteredLine.split(",")
                    target = int(target)
                    value = float(value)
                    if value >= 0.0:
                        if self.rwdPrecision:
                            value = round(value,self.rwdPrecision)
                        if target not in self.targetValues or value > self.targetValues[target]:
                            self.targetValues[target] = value
        print(f"\nreadPreFireTargetValues() file: {filepath}, count: {len(self.targetValues)}")


    def readActiveFireTargetValues(self):
        activeFirePath = self.activeFireRewardsPath

        filepath = f"{self.dataPath}{activeFirePath}TV_ACTIVE_FIRE.csv"
        assert os.path.exists(filepath), f"\nreadActiveFireTargetValues() ERROR! file not found {filepath}"

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.strip().split(",")
                target = int(row[0])
                value = float(row[1])
                if value > 0.0:
                    if self.rwdPrecision:
                        value = round(value, self.rwdPrecision)
                    self.activeFireTargets[target] = value
                    if target not in self.targetValues or value > self.targetValues[target]:
                        self.targetValues[target] = value
        print(f"readActiveFireTargetValues() file: {filepath}, count: {len(self.activeFireTargets)}\n")


    def increaseTargetValues(self):
        minActiveTargetRwd = 50000
        maxPreTargetRwd = 0
        minActiveTargetRwdAdjusted = 50000
        maxPreTargetRwdAdjusted = 0
        for target in self.targetValues:
            if target in self.activeFireTargets:
                if self.targetValues[target] < minActiveTargetRwd:
                    minActiveTargetRwd = self.targetValues[target]
                self.targetValues[target] = 1000 * self.targetValues[target]
                if self.targetValues[target] < minActiveTargetRwdAdjusted:
                    minActiveTargetRwdAdjusted = self.targetValues[target]
            else:
                if self.targetValues[target] > maxPreTargetRwd:
                    maxPreTargetRwd = self.targetValues[target]
                self.targetValues[target] = 100 * self.targetValues[target]
                if self.targetValues[target] > maxPreTargetRwdAdjusted:
                    maxPreTargetRwdAdjusted = self.targetValues[target]
        print("\nMin active target rwd: "+str(minActiveTargetRwd)+", max pre target rwd: "+str(maxPreTargetRwd)+"\n")
        print("\nMin active target rwd (adjusted): "+str(minActiveTargetRwdAdjusted)+", max pre target rwd (adjusted: "+str(maxPreTargetRwdAdjusted)+"\n")


    def removeLowValueChoices(self):
        removedTargets = set()
        removedSpecPts = {}
        removedTicks = {}
        removedTickCount = 0
        for sat in self.satChoices:
            filteredSatChoices = {}
            choices = self.satChoices[sat]
            for tick in choices:
                tickChoices = choices[tick]
                if "DNL" in tickChoices:
                    filteredSatChoices[tick] = tickChoices
                else:
                    filteredTargetChoices = {}
                    for specularPt in tickChoices:
                        filteredSpecularPtTargets = []
                        specChoices = tickChoices[specularPt]
                        for target in specChoices:
                            rwd = self.targetValues[target]
                            if rwd > self.rwdThreshold:
                                filteredSpecularPtTargets.append(target)
                            else:
                                removedTargets.add(target)
                        if filteredSpecularPtTargets:
                            filteredTargetChoices[specularPt] = filteredSpecularPtTargets
                        else:
                            if sat not in removedSpecPts:
                                removedSpecPts[sat] = []
                            removedSpecPts[sat].append({"tick": tick, "specPt": specularPt})
                    if filteredTargetChoices:
                        filteredSatChoices[tick] = filteredTargetChoices
                    else:
                        if sat not in removedTicks:
                            removedTicks[sat] = []
                        removedTicks[sat].append(tick)
                        removedTickCount += 1
            self.satChoices[sat] = filteredSatChoices
        skippedTargetRwds = [self.targetValues[x] for x in removedTargets]
        totalSkippedRwd = round(np.sum(skippedTargetRwds),3)
        avgSkippedRwd = round(np.average(skippedTargetRwds),3)
        maxSkippedRwd = round(np.max(skippedTargetRwds),3)
        print("\nThreshold filter "+str(self.rwdThreshold)+":  Removed "+str(len(skippedTargetRwds))+" targets, total skipped rwds: "+str(totalSkippedRwd)+", avg skipped rwd: "+str(avgSkippedRwd)+", max: "+str(maxSkippedRwd)+", removed images: "+str(removedTickCount)+"\n")

    def removeZeroValueChoices(self):
        zeroValTargets = []
        nzTargets = []
        for target in self.targetValues:
            val = self.targetValues[target]
            if val <= 0.0:
                zeroValTargets.append(target)
            else:
                nzTargets.append(target)
        for zTarget in zeroValTargets:
            if zTarget in self.targetValues:
                self.targetValues.pop(zTarget)
        print("\nNon-Zero filter  Removed "+str(len(zeroValTargets))+" targets, nzTargets: "+str(len(nzTargets))+"\n")

    def multiHisto(self):
        # Generate some data for the two bars
        expA = None
        expB = None
        with open("allRewards.expA.histo.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                expA = ast.literal_eval(line)
                break
        with open("allRewards.expB.histo.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                expB = ast.literal_eval(line)
                break
        # Create the figure and axes
        fig, ax = plt.subplots()

        # Plot the histograms
        ax.hist([expA, expB], bins=28, label=['Experiment A', 'Experiment B'], rwidth=5,edgecolor='white', linewidth=1, color=['blue', 'red'])

        # xtix = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,6,6.5,7,7.5]
        # plt.xticks(xtix)

        current_values = plt.gca().get_yticks()
        # using format string '{:.0f}' here but you can choose others
        # yLabels = [str(int(x/1000))+"k" for x in current_values[1:]]
        # yLabels.insert(0,"0")
        # plt.gca().set_yticklabels(yLabels)

        yTix = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000]
        # yTix = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000]
        plt.yticks(yTix)
        # Add labels and title
        ax.set_xlabel('Target Reward')
        ax.set_ylabel('Target Count')
        # ax.set_title('Reward Histogram')
        plt.grid(True,"major","y")
        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

    def createRewardHistogramNew(self, file=None):
        if file:
            targets = self.readSelectedTargetsFromFile(file)
            allRewards = [self.targetValues[target] for target in targets]
        else:
            allRewards = [self.targetValues[target] for target in self.targetTimes]
            with open("allRewards."+self.experiment+".txt", "w") as f:
                f.write(str(allRewards))
        sumRwd = sum(allRewards)
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = round(np.average(allRewards),2)
        stdDevRwd = round(np.std(allRewards),2)
        print("\nAll Target Reward Summary: Target count: "+str(len(allRewards))+", Available rewards: "+str(sumRwd)+", Min rwd: "+str(minRwd)+", Max rwd: "+str(maxRwd)+", avg: "+str(avgRwd)+", stdDev: "+str(stdDevRwd))
        bins = []
        for i in range(7):
            bins.append(i)
            bins.append(i+0.25)
            bins.append(i+0.5)
            bins.append(i+0.75)
        bins.append(7)
        counts, edges, patches = plt.hist(allRewards, bins=bins, edgecolor='white', linewidth=1)
        maxCount = int(max(counts)+1)

        # labels = list('abcdefghijklmnopqrstuvwxyz')
        #
        # allYticks = [x for x in range(0,maxCount+1,1000)]
        #
        # def format_fn(tick_val, tick_pos):
        #     v = None
        #     for t in allYticks:
        #         if int(tick_val) == t:
        #             v = t
        #             break
        #
        #     if v:
        #         return str(v)[0]+"K"
        #     else:
        #         return ''

        xtix = [x for x in bins[::4]]
        # xtix.append(8)
        ytix = [y/1000 for y in range(0, maxCount+1,100)]
        plt.xticks(xtix)
        # plt.yticks(allYticks)
        title = "Experiment A" #Target Values Histogram\n  exp3, threshold: 0.2, \n *OPTIMAL* Objective: 19,568"
        plt.title(title)
        plt.xlabel('Target Value')
        plt.ylabel('Target Count (Thousands)')

        current_values = plt.gca().get_yticks()
        # using format string '{:.0f}' here but you can choose others
        yLabels = [str(int(x/1000))+"k" for x in current_values[1:]]
        yLabels.insert(0,"0")
        plt.gca().set_yticklabels(yLabels)

        # fig, ax = plt.subplots()
        # ax.yaxis.set_major_formatter(format_fn)
        plt.show()

    def plotTest(self):
        fig, ax = plt.subplots()
        xs = range(26)
        ys = range(26)
        labels = list('abcdefghijklmnopqrstuvwxyz')

        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xs:
                return labels[int(tick_val)]
            else:
                return ''

        # A FuncFormatter is created automatically.
        ax.xaxis.set_major_formatter(format_fn)
        # from matplotlib.ticker import MaxNLocator
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(xs, ys)
        plt.show()

    def createRewardHistogram(self, file=None):
        if file:
            targets = self.readSelectedTargetsFromFile(file)
            allRewards = [self.targetValues[target] for target in targets]
        else:
            allRewards = [self.targetValues[target] for target in self.targetTimes]
        sumRwd = sum(allRewards)
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = round(np.average(allRewards),2)
        stdDevRwd = round(np.std(allRewards),2)
        print("\nAll Target Value Summary: Targets: "+str(len(allRewards))+", sum: "+str(sumRwd)+", min: "+str(minRwd)+", max: "+str(maxRwd)+", avg: "+str(avgRwd)+", stdDev: "+str(stdDevRwd))
        bins = []
        for i in range(7):
            bins.append(i)
            bins.append(i+0.25)
            bins.append(i+0.5)
            bins.append(i+0.75)
        bins.append(7)
        counts, edges, patches = plt.hist(allRewards, bins=bins, edgecolor='white', linewidth=1)
        maxCount = int(max(counts)+1)
        xtix = [x for x in bins[::2]]
        # xtix.append(8)
        ytix = [y for y in range(0, maxCount+1,100)]
        plt.xticks(xtix)
        # plt.yticks(ytix)
        title = "Target Values Histogram\n  exp3, threshold: 0.2, \n *OPTIMAL* Objective: 19,568"
        plt.title(title)
        plt.xlabel('Target Value')
        plt.ylabel('Target Count')
        plt.show()

    def createResidualRewardHistogram(self, file=None):
        allVisibleTargets = list(self.targetTimes.keys())
        if file:
            selectedTargets = self.readSelectedTargetsFromFile(file)
        else:
            selectedTargets = self.selectedTargets
        remainingTargets = []
        capturedTargets = []
        for target in allVisibleTargets:
            if target not in selectedTargets:
                remainingTargets.append(target)
            else:
                capturedTargets.append(target)

        allRewards = [self.targetValues[target] for target in remainingTargets]
        sumRwd = sum(allRewards)
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = round(np.average(allRewards),2)
        stdDevRwd = round(np.std(allRewards),2)
        print("\nResidual Target Value Summary: Targets: "+str(len(allRewards))+", sum: "+str(sumRwd)+", min: "+str(minRwd)+", max: "+str(maxRwd)+", avg: "+str(avgRwd)+", stdDev: "+str(stdDevRwd))
        bins = []
        for i in range(7):
            bins.append(i)
            bins.append(i+0.25)
            bins.append(i+0.5)
            bins.append(i+0.75)
        bins.append(7)
        counts, edges, patches = plt.hist(allRewards, bins=bins, edgecolor='white', linewidth=1)
        maxCount = int(max(counts)+1)
        xtix = [x for x in bins[::2]]
        # xtix.append(8)
        ytix = [y for y in range(0, maxCount+1,100)]
        plt.xticks(xtix)
        # plt.yticks(ytix)
        title = "Residual Target Values Histogram\n  exp3, threshold: 0.20, 6-hr limit\n Objective: 20271.7,    Gap: 0.08,   Time: 24-hrs"
        plt.title(title)
        plt.xlabel('Target Value')
        plt.ylabel('Target Count')
        plt.show()


    def showSelectedTargetRewardHistogram(self):
        allRewards = [(self.targetValues[target]) for target in self.selectedTargets]
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = np.average(allRewards)
        stdDevRwd = np.std(allRewards)
        print("\nSelected Target Value Summary: Targets: "+str(len(allRewards))+", min: "+str(minRwd)+", max: "+str(maxRwd)+", avg: "+str(avgRwd)+", stdDev: "+str(stdDevRwd))

        bins = []
        for i in range(8):
            bins.append(i)
            bins.append(i+0.1)
            bins.append(i+0.2)
            bins.append(i+0.3)
            bins.append(i+0.4)
            bins.append(i+0.5)
            bins.append(i+0.6)
            bins.append(i+0.7)
            bins.append(i+0.8)
            bins.append(i+0.9)
        counts, edges, patches = plt.hist(allRewards, bins=bins, edgecolor='white', linewidth=1)
        maxCount = int(max(counts)+1)
        xtix = [x for x in bins[::2]]
        xtix.append(8)
        ytix = [y for y in range(0, maxCount+1,100)]
        plt.xticks(xtix)
        # plt.yticks(ytix)
        title = "Target Value Histogram, experiment: "+str(self.experiment)+", threshold: "+str(self.rwdThreshold)
        plt.title(title)
        plt.xlabel('Target Value')
        plt.ylabel('Target Count')
        plt.show()
        plt.show()

    def createDataCycles(self):
        self.imageCount = 0
        dnlId = 1
        for satId in self.satChoices:
            cycles = []
            cycle = {"obs": [], "dnl": []}
            dnlWindow = None
            commandChoices = self.satChoices[satId]
            timepoints = sorted(commandChoices)
            cycles.append(cycle)
            for cmdTime in timepoints:
                cmd = commandChoices[cmdTime][0]
                if cmd['cmd'] == "obs": #self.isObsCmd(cmd):
                    if cycle["dnl"]:
                        # if current cycle contains downlinks, then end it, and start a new cycle
                        cycle = {"obs": [], "dnl": []}
                        cycles.append(cycle)
                    self.imageCount += 1
                    targets = self.getObsTargets(cmd)
                    self.setTargetTimes(satId, targets)
                    image = {"sat": satId, "time": cmdTime, "cycle": len(cycles) - 1, "image": self.imageCount,
                             "targets": targets, "type": "large"}
                    cycle["obs"].append(image)
                    self.allImages.append(image)
                    targetCount = len(targets)
                    # if self.includeSensorModes and targetCount > 5:
                    #     self.imageCount += 1
                    #     smallTargets = [target for target in targets]
                    #     while len(smallTargets) > targetCount/2:
                    #         smallTargets = smallTargets[1:-1]
                    #     # self.setTargetTimes(satId, smallTargets)
                    #     image = {"sat": satId, "time": cmdTime, "cycle": len(cycles) - 1, "image": self.imageCount,
                    #              "targets": smallTargets, "type": "small"}
                    #     cycle["obs"].append(image)
                    #     self.allImages.append(image)


                elif cycle["obs"]: # only collect dnls after obs in cycle
                    gs = list(cmd['targets'])[0]
                    if not dnlWindow:
                        # create first dnlWindow for this sat
                        dnlWindow = {"start": cmdTime, "end": cmdTime, "gs": gs, "dnlId": dnlId}
                        cycle["dnl"].append(dnlWindow)
                    elif cmdTime == dnlWindow["end"] + 1 and gs == dnlWindow["gs"]:
                        dnlWindow["end"] = cmdTime
                    else:
                        # start a new dnlWindow
                        dnlId += 1
                        dnlWindow = {"start": cmdTime, "end": cmdTime, "gs": gs, "dnlId": dnlId}
                        cycle["dnl"].append(dnlWindow)

            self.allSatCycles[satId] = cycles
        self.addCycleStartAndEndTimes()

    def setTargetTimes(self, satId, targets):
        for target in targets:
            if target not in self.targetTimes:
                self.targetTimes[target] = {}
            if satId not in self.targetTimes[target]:
                self.targetTimes[target][satId] = []
            self.targetTimes[target][satId].append(self.imageCount)

    def isObsCmd(self, cmd):
        return "DNL" not in cmd

    def getObsTargets(self, cmd):
        targets = cmd['targets']
        return sorted(targets)

    def getDnlTarget(self, cmd):
        return cmd["DNL"]

    def addCycleStartAndEndTimes(self):
        for satId in self.allSatCycles:
            cycleStart = 0
            cycles = self.allSatCycles[satId]
            for cycle in cycles:
                cycleEnd = self.getCycleEndTime(cycle)
                cycle["start"] = cycleStart
                cycle["end"] = cycleEnd
                cycleStart = cycleEnd+1

    def getCycleDuration(self, cycle):
        dur = cycle["end"] - cycle["start"] + 1
        return dur

    def getMaxCycleDurationNoEclipse(self):
        maxDur = 0
        for sat in self.allSatCycles:
            for cycle in self.allSatCycles[sat]:
                cycleEclipseDur = self.getEclipseTickCount(sat, cycle["start"], cycle["end"])
                cycleDur = self.getCycleDuration(cycle)
                dur = cycleDur - cycleEclipseDur
                if dur > maxDur:
                    maxDur = dur
        return maxDur

    def writeLatencies(self):
        filepath = f"{self.dataPath}{self.plannerOutputPath}"
        filename = filepath + f"latencies.txt"
        with open(filename, "w") as f:
            for sat in self.allSatCycles:
                f.write("\nSatellite "+sat+":\n")
                cycleId = 0
                f.write("\n")
                for cycle in self.allSatCycles[sat]:
                    latencies = cycle["latencies"] if "latencies" in cycle else []
                    f.write("   cycle " + str(cycleId) + ", "+str(len(latencies))+" images downlinked: :\n")
                    if latencies:
                        for image in cycle["latencies"]:
                            f.write(str(image)+"\n")
                    cycleId += 1



    def writeSelectedTargets(self):
        filename = "selectedTargets." + self.planExecutionDate + "." + self.targetValFile[3:-4] + ".txt"
        with open(filename, "w") as f:
            msg = "# Selected target count: "+str(len(self.selectedTargets))
            print(msg)
            f.write(msg)
            f.write("\n\n"+str(self.selectedTargets))


    def reportSelectedTargets(self, file):
        missingActiveTargets = []
        selectedActiveTargets = []
        selectedPreTargets = []
        for target in self.selectedTargets:
            if target in self.activeFireTargets:
                selectedActiveTargets.append(target)
            else:
                selectedPreTargets.append(target)
        for activeTarget in self.activeFireTargets:
            if activeTarget not in self.selectedTargets:
                missingActiveTargets.append(activeTarget)
        selectedActiveFireCount = len(selectedActiveTargets)
        selectedPreFireCount = len(selectedPreTargets)
        allTargetCount = len(self.targetTimes)
        availActiveFireCount = len(self.activeFireTargets)-len(self.unavailableActiveFireTargets)
        availPreFireCount = allTargetCount - availActiveFireCount
        selectedPrePct = round((selectedPreFireCount/availPreFireCount) * 100, 2)
        selectedActivePct = round((selectedActiveFireCount/availActiveFireCount) * 100, 2) if availActiveFireCount else 0
        selectedTargetRewards = 0
        for target in self.selectedTargets:
            selectedTargetRewards += self.targetValues[target]
        selectedTargetRewards = round(selectedTargetRewards, 2)
        selectedRewardPct = round((selectedTargetRewards/self.totalAvailableRewards) * 100,2)
        msg = f"\nSelected Target Rewards: {selectedTargetRewards}/{round(self.totalAvailableRewards,2)} ({selectedRewardPct} %)"
        print(msg)
        file.write(msg+"\n")
        msg = f"Selected Available Active Fire Targets: {selectedActiveFireCount}/{availActiveFireCount} ({selectedActivePct} %)"
        print(msg)
        file.write(msg+"\n")
        msg = f"Selected Pre-Fire Targets: {selectedPreFireCount}/{availPreFireCount}  ({selectedPrePct} %)"
        print(msg)
        file.write(msg+"\n")
        if missingActiveTargets:
            msg = f"\nUnavailable Active Fire Targets ({len(self.unavailableActiveFireTargets)}):\n{self.unavailableActiveFireTargets}"
            print(msg)
            file.write(msg + "\n")
            msg = f"Unselected Active Fire Targets ({len(missingActiveTargets)}):\n{missingActiveTargets}"
            print(msg)
            file.write(msg + "\n")

    # def writeActiveFireTargets(self):
    #     filename = "activeFireTargets."+self.planExecutionDate+".txt"
    #     nzTargets = []
    #     for target in self.targetValues:
    #         if self.targetValues[target] > 0.0:
    #             nzTargets.append(target)
    #     with open(filename, "w") as f:
    #         f.write("# Active Fire Targets "+self.planExecutionDate+" ("+str(len(nzTargets))+")")
    #         f.write("\n\n"+str(nzTargets))

    def printRewardStats(self, file=None):
        targets = [x for x in self.targetTimes if not isinstance(x, str)]
        allRewards = [self.targetValues[target] for target in targets]
        self.totalAvailableRewards = round(sum(allRewards), 2)
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = round(np.average(allRewards), 2)
        stdDevRwd = round(np.std(allRewards), 2)
        msg = "Available Target Rewards: " + str(
            self.totalAvailableRewards) + ", Min rwd: " + str(minRwd) + ", Max rwd: " + str(maxRwd) + ", avg: " + str(avgRwd) + ", stdDev: " + str(
            stdDevRwd)
        print(msg)
        if file:
            file.write(msg)

    def printCyclesAndCollectFinalStates(self, solverResult=None):
        filepath = f"{self.dataPath}{self.plannerOutputPath}planSummary.txt"
        with open(filepath, "w") as f:
            msg = "Data Cycles:"
            print(msg)
            f.write(msg+ '\n')
            imageCount = 0
            self.finalSatStates = {}
            for satId in self.allSatCycles:
                self.finalSatStates[satId] = {"storage": None, "energy": None}
                cycles = self.allSatCycles[satId]
                msg = "\n-------------\nSat: " + str(satId) + ", cycles (" + str(len(cycles)) + "):"
                print(msg)
                f.write(msg+'\n')
                cycleId = 0
                for cycle in cycles:
                    obs = cycle["obs"]
                    dnl = cycle["dnl"]
                    selectedImages = cycle["selectedImages"] if "selectedImages" in cycle else None
                    firstObsTime = obs[0]["time"] if obs else None
                    lastObsTime = obs[-1]["time"] if obs else None
                    firstImage, lastImage = self.firstAndLastImagesInCycle(cycle)
                    lastImage = obs[-1]["image"] if obs else None
                    firstDnl = dnl[0] if dnl else None
                    lastDnl = dnl[-1] if dnl else None
                    obsPct = 0
                    dnlPct = 0
                    if firstObsTime and lastObsTime:
                        obsTicks = len(obs)
                        obsDur = lastObsTime - firstObsTime + 1
                        obsPct = round((obsTicks/obsDur) * 100.0,2)
                    else:
                        obsDur = ""
                        obsTicks = None
                    if firstDnl and lastDnl:
                        dnlDur = lastDnl["end"] - firstDnl["start"] + 1
                        dnlTicks = self.getDnlTickCount(cycle) #len(dnl)
                        if dnlTicks and dnlDur:
                            dnlPct = round((dnlTicks/dnlDur) * 100.0, 2)
                    else:
                        dnlDur = ""
                        dnlTicks = None
                    if lastObsTime and firstDnl:
                        assert lastObsTime < firstDnl["start"], "printCycles() ERROR! obsTime "+str(lastObsTime)+" > dnlTime "+str(firstDnl)
                    msg = "\n  Cycle "+str(cycleId)+" ["+str(cycle["start"])+" - "+str(cycle["end"])+"] "
                    msg += "    Obs: "+str(firstObsTime)+" - "+str(lastObsTime)+" ("+str(obsTicks)+"/"+str(obsDur)+" = "+str(obsPct)+" %), images: "+str(firstImage)+"-"+str(lastImage)
                    if selectedImages:
                        for imageId in selectedImages:
                            image = self.getImage(imageId)
                            imageCount += 1
                    if firstDnl and lastDnl:
                        msg += "\n      DNL: "+str(firstDnl["start"])+" - "+str(lastDnl["end"]) + " ("+str(dnlTicks)+"/"+str(dnlDur)+" = "+str(dnlPct) +" %),  GS ("+str(len(cycle["dnl"]))+"):"
                        for slot in cycle["dnl"]:
                            dur = slot["end"] - slot["start"] + 1
                            msg += " ["+slot["gs"]+": "+str(slot["start"]) + " - "+str(slot["end"])+" ("+str(dur)+")]"
                    print(msg)
                    f.write(msg + '\n')
                    if "dnlPlan" in cycle:
                        dnlPlan = cycle["dnlPlan"]
                        dnlSlots = []
                        dnlAssignments = []
                        for z in dnlPlan["z"]:
                            if z[0] == satId and z[1] == cycleId:
                                dnlSlots.append(z)
                        for dnlSlot in dnlSlots:
                            gs = dnlSlot[2]
                            n = dnlSlot[3]
                            tTime = 0
                            for t in dnlPlan["t"]:
                                if t[0] == dnlSlot:
                                    tTime = t[1]
                            pTime = 0
                            for p in dnlPlan["p"]:
                                if p[0] == dnlSlot:
                                    pTime = p[1]
                            assignment = {"g": gs, "n": n, "t": tTime, "p": pTime }
                            dnlAssignments.append(assignment)
                        msg = "         DNL Plan: "
                        totalDnlDur = 0
                        for x in dnlAssignments:
                            tTime = x["t"]
                            pTime = x["p"]
                            if tTime > 0.0 and pTime > 0.0:
                                endTime = tTime + pTime -1
                                dur = pTime
                                totalDnlDur += dur
                                msg += " [" + x["g"] + ": " + str(tTime) + " - " + str(endTime)+" ("+str(dur)+")]"
                        msg += " Total: "+str(totalDnlDur)
                        print(msg)
                        f.write(msg + '\n')

                    if "availSpace" in cycle:
                        # TODO: BUG? should this be prior cycle data vs. current?
                        varIndex = "[" + satId + ", " + str(cycleId) + "]"
                        sa = round(cycle["availSpace"],2)
                        su = round(cycle["usedSpace"],2)
                        sf = round(cycle["freedSpace"],2)
                        saFinal = round(sa - su + sf,2)
                        downlinkTicks = round(sf / self.dnlRate,2)
                        if self.includeEnergyConstraints:
                            ea = round(cycle["energyAvail"],2)
                            # eRaw = round(cycle["energyRaw"],2)
                            eNet = round(cycle["energyNet"],2)
                            powerIn = round(cycle["powerIn"],2)
                            powerOut = round(cycle["powerOutDefault"],2)
                            powerOutDnl = round(downlinkTicks * self.powerModel["powerOutDnlPct"],2)
                        selectedImageCount = len(selectedImages) if selectedImages else 0
                        dnlPlanSecs = cycle["dnlPlanSecs"] if "dnlPlanSecs" in cycle else 0
                        msg =     "      Plan: selected images: "+str(selectedImageCount)+", downlink secs: "+str(downlinkTicks) +", [dnlPlanSecs: "+str(dnlPlanSecs)+"]"
                        msg += "\n        sAvail "+str(sa)+" - sUsed "+str(su)+" + sFreed "+str(sf)+" = "+str(saFinal)
                        if self.includeEnergyConstraints:
                            msg += "\n        eAvail " +str(ea)
                            msg += "\n        eNet "+str(eNet) +" = eAvail " +str(ea)+" +  eIn " +str(powerIn)+" - eOut "+str(powerOut) + " - eOutDnl "+ str(powerOutDnl)
                            # msg += "\n        eRaw "+str(eRaw) +" = eAvail " +str(ea)+" + eNet "+str(eNet)

                        print(msg)
                        f.write(msg + '\n')
                        self.finalSatStates[satId]["storage"] = saFinal
                        self.finalSatStates[satId]["energy"] = min(eNet, 100)
                    cycleId += 1
            allTargetCount = len(self.targetTimes)
            activeFireCount = len(self.activeFireTargets)
            preFireCount = allTargetCount - activeFireCount
            msg = f"\n\n------------\nAvailable target counts: {preFireCount} (pre) + {activeFireCount} (active) = {allTargetCount} total"
            print(msg)
            f.write(msg + '\n')
            self.printRewardStats(file=f)
            if solverResult:
                msg = f"\n-----------\nSolver results\n Status: {solverResult['status']}, Objective: {solverResult['objective']}, MIP Gap: {solverResult['gap']}, Solve time: {solverResult['solveTime']}"
                print(msg)
                f.write(msg + '\n')
            self.reportSelectedTargets(f)
            msg = f"\nSelected image count: {imageCount}"
            print(msg)
            f.write(msg + '\n')

    def availableDownlinkSecsInWindow(self, sat, checkpoint1, checkpoint2):
        downlinkSecs = []
        dataCycles = self.allSatCycles[sat]
        for cycle in dataCycles:
            if checkpoint2 < cycle["start"] or checkpoint1 > cycle["end"]:
                continue
            elif "dnl" in cycle:
                for tick in cycle["dnl"]:
                    if checkpoint1 <= tick and tick <= checkpoint2:
                        downlinkSecs.append(tick)
        return len(downlinkSecs)


    def collectDownlinkWindows(self):
        self.downlinkWindows = {}
        for sat in self.satChoices:
            if sat not in self.downlinkWindows:
                self.downlinkWindows[sat] = []
            satChoices = self.satChoices[sat]
            priorDnlChoice = None
            currentDnlWindow = {}
            for tick in satChoices:
                choiceDict = satChoices[tick]
                for key in choiceDict:
                    if key == "DNL":
                        gs = choiceDict[key]
                        if not currentDnlWindow:
                            currentDnlWindow = {"sat": sat, "gs": gs, "start": tick, "end": tick, "dur": 0}
                        else:
                            if currentDnlWindow and currentDnlWindow["gs"] == gs and currentDnlWindow["end"]+1 == tick:
                                currentDnlWindow["end"] = tick
                                dur = tick - currentDnlWindow["start"]+1
                                currentDnlWindow["dur"] = dur
                            else:
                                self.downlinkWindows[sat].append(currentDnlWindow)
                                currentDnlWindow = {"sat": sat, "gs": gs, "start": tick, "end": tick, "dur": 0}

    def findDownlinkConflicts(self):
        for sat1 in self.downlinkWindows:
            for w1 in self.downlinkWindows[sat1]:
                for sat2 in self.downlinkWindows:
                    if sat1 != sat2:
                        for w2 in self.downlinkWindows[sat2]:
                            conflict = self.isDownlinkConflict(w1,w2)
                            if conflict:
                                self.downlinkConflicts.append(conflict)
        if self.downlinkConflicts:
            print("\nDownlink Conflicts ("+str(len(self.downlinkConflicts))+")")
            totalDuration = 0
            maxDuration = None
            for conflict in self.downlinkConflicts:
                if not maxDuration or conflict["dur"] > maxDuration["dur"]:
                    maxDuration = conflict
                totalDuration += conflict["dur"]
                msg = conflict["sat1"]+", "+conflict["sat2"]+": "+conflict["gs"] +" "+str(conflict["start"])+" - "+str(conflict["end"])+" ("+str(conflict["dur"])+")"
                msg += "\n    "+str(conflict["w1"])
                msg += "\n    "+str(conflict["w2"])
                print(msg+"\n")
            avgDur = round(totalDuration/len(self.downlinkConflicts),3)
            print("Total conflict duration: "+str(totalDuration)+", avg: "+str(avgDur)+", max: "+str(maxDuration))

    def isDownlinkConflict(self, w1, w2):
        if w1["gs"] != w2["gs"] or w2["start"] > w1["end"] or w1["start"] > w2["end"]:
            return None
        else:
            overlapStart = max(w1["start"], w2["start"])
            overlapEnd = min(w1["end"], w2["end"])
            dur = overlapEnd - overlapStart + 1
            result = {"sat1": w1["sat"], "sat2": w2["sat"], "gs": w1["gs"], "start": overlapStart, "end": overlapEnd, "dur": dur, "w1": w1, "w2": w2}
            return result

    def firstAndLastImagesInCycle(self, cycle):
        obs = cycle["obs"]
        first = obs[0]["image"] if obs else None
        last =  obs[-1]["image"] if obs else None
        return first, last

    def lastImageInSatCycle(self, sat, cycle):
        cycle = self.allSatCycles[sat][cycle]
        obs = cycle["obs"]
        return obs[-1]["image"] if obs else None

    def imagesContainingTarget(self, target):
        targetOpportunities = self.targetTimes[target]
        images = []
        for sat in targetOpportunities:
            images.extend(targetOpportunities[sat])
        return images

#  Power model

    def readEclipseFileForSat(self, satId):
        print("\nreadEclipseFilesForSat() sat: "+str(satId))
        if satId not in self.eclipses:
            self.eclipses[satId] = set()
        satEclipses = self.eclipses[satId]
        path = f"{self.dataPath}{self.orbitsPath}{satId}/eclipse/"
        # path = f"{self.dataPath}orbits/sample/output/{satId}/eclipse/"
        # path = "./inputs/"+self.experiment+"/eclipses/"
        assert os.path.exists(path), "readEclipseFileForSat() ERROR! path not found: "+path
        eclipseFiles = [f for f in os.listdir(path) if "eclipse" in f]
        # TODO: Is there only one eclipseFile per sat?
        for file in eclipseFiles:
            # if satId in file:
            if True: #satId in file:
                filepath = path + file
                print("reading eclipse  file: "+filepath)
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line.startswith("start"):
                            if line.count(",") > 0:
                                terms = line.split(",")
                                start = int(terms[0])
                                end   = int(terms[1])
                                # satEclipses.append((start, end))
                                eclipse = [x for x in range(start, end+1)]
                                satEclipses.update(eclipse)
                                if satId not in self.energyCheckpoints:
                                    self.energyCheckpoints[satId] = [0]
                                self.energyCheckpoints[satId].append(end)


    def initPowerModel(self):
        self.readPowerConfigFile()
        # power model constants
        self.energyMax = self.powerModel["maxCharge"] * 3600 # Joules
        self.energyMin     = self.energyMax * (self.powerModel["minChargePct"]/100)     # Joules
        self.initialEnergy = self.energyMax * (self.powerModel["initialChargePct"]/100) # Joules
        print("\ninitPowerModel() model: "+str(self.powerModel) +" initial: "+str(self.initialEnergy)+", min: "+str(self.energyMin)+", max: "+str(self.energyMax)+"\n")

    def readPowerConfigFile(self):
        print("readPowerConfigFile()")
        path = f"{self.dataPath}planner/powerConfig.txt"
        assert os.path.exists(path), "readPowerConfigFile() ERROR! path not found: "+path
        with open(path, "r") as f:
            dictIn = ""
            for rawLine in f:
                line = rawLine.strip()
                if line and not line.startswith("#"):
                    dictIn += line
        dictIn = dictIn.strip()
        self.powerModel = ast.literal_eval(dictIn)
        print ("Power config: "+str(self.powerModel))
        # self.powerModel = powerConfig[self.powerModel])

    def readSelectedTargetsFromFile(self, file):
        with open(file, "r") as f:
            skipLine = True
            targetString = ""
            for line in f:
                if line.startswith("# selected"):
                    skipLine = False
                elif not skipLine:
                    targetString += line.strip()
            targets = ast.literal_eval(targetString)
        return targets

    def collectSatObsPlanForCycle(self, sat, cycle):
        cycleImages = []
        for obs in self.satPlans[sat]:
            if obs["cycle"] == cycle:
                cycleImages.append(self.getImage(obs["image"]))
        return cycleImages

    # def calculateSatLatencies(self, sat):
    #     print("\nSatellite: "+sat)
    #     allLatencies = []
    #     buffer = []
    #     cycleId = -1
    #     collectedImageCount = 0
    #     for cycle in self.allSatCycles[sat]:
    #         cycleCollectedImageCount = len(cycle["selectedImages"]) if "selectedImages" in cycle else 0
    #         collectedImageCount += cycleCollectedImageCount
    #         cycleId +=1
    #         cycleImages = self.collectSatObsPlanForCycle(sat, cycleId)
    #         bufferSize = len(buffer)
    #         totalImages = bufferSize + cycleCollectedImageCount
    #         print("\n============\n\nSatellite: "+sat+", Cycle "+str(cycleId)+": "+ str(bufferSize)+" images in buffer + " +str(cycleCollectedImageCount)+ " images collected = "+str(totalImages))
    #         print("| Image | Collected | Downlinked | Latency | Remaining Images |")
    #         if "selectedImages" in cycle:
    #             for imageId in cycle["selectedImages"]:
    #                 time = self.getImage(imageId)["time"]
    #                 buffer.append(
    #                     {"id": imageId, "collectionTime": time, "remainingPct": 100})
    #         if buffer:
    #             dnlPlan = cycle["dnlPlan"] if "dnlPlan" in cycle else None
    #             if dnlPlan:
    #                 currentImage = buffer[0]
    #                 # print("Current image: sat " + str(sat) + ", cycle " + str(cycleId) + ", " + str(
    #                 #     currentImage)+", imageCount: "+str(len(buffer)))
    #                 currentImageCollectionTime = currentImage["collectionTime"]
    #                 for dnlWindowIndex in range(len(cycle["dnl"])): #dnlPlanTimes)):
    #                     dWindow = cycle["dnl"][dnlWindowIndex]
    #                     if buffer and currentImage:
    #                         dnlStartTime = dWindow["start"] #int(dnlPlanTimes[dnlWindowIndex][1])
    #                         dnlEndTime = dWindow["end"]
    #                         if currentImageCollectionTime < dnlEndTime:
    #                             for tick in range(dnlStartTime, dnlEndTime+1):
    #                                 # subtract 5 % image remaining per second (20 seconds to downlink an image)
    #                                 if currentImageCollectionTime <= tick:
    #                                     currentImage["remainingPct"] = round(currentImage["remainingPct"] - 5, 3)
    #                                     if currentImage["remainingPct"] <=0:
    #                                         currentImage["downlinkTime"] = tick
    #                                         latency = round((currentImage["downlinkTime"] - currentImage["collectionTime"] + 1)/60, 2)
    #                                         assert latency > 0, "calcualteSatLatencies() ERROR! Negative Latency: sat: "+str(sat)+", cycle: "+str(cycle)+", image: "+str(currentImage)+", latency: "+str(latency)
    #                                         currentImage["latency"] = latency
    #                                         allLatencies.append(currentImage["latency"])
    #                                         if "latencies" not in cycle:
    #                                             cycle["latencies"] = []
    #                                         downlinkedImage = buffer.pop(0)
    #                                         cycle["latencies"].append(downlinkedImage)
    #                                         imageInfo = str(downlinkedImage["id"])+", collected: "+str(downlinkedImage["collectionTime"])+", downlinked: "+str(downlinkedImage["downlinkTime"])+", latency: "+str(downlinkedImage["latency"])
    #                                         imageTableInfo = "| " +str(downlinkedImage["id"]).rjust(5," ")+" | "+str(downlinkedImage["collectionTime"]).rjust(9, " ")+" | "+str(downlinkedImage["downlinkTime"]).rjust(10, " ")+" | "+str(downlinkedImage["latency"]).rjust(7, " ")+" | "+str(len(buffer)).rjust(8," ")+"        |"
    #                                         print(imageTableInfo)
    #                                         if buffer:
    #                                             currentImage = buffer[0]
    #                                             currentImageCollectionTime = currentImage["collectionTime"]
    #                                         else:
    #                                             currentImage = None
    #                                             print("Buffer Empty! sat " + str(sat) + ", cycle " + str(cycleId)+"\n")
    #                                             break
    #         # end for cycle
    #     if allLatencies:
    #         avg = str(round(np.average(allLatencies),2))
    #         std = str(round(np.std(allLatencies),2))
    #         mx = str(round(np.max(allLatencies),2))
    #         mn = str(round(np.min(allLatencies),2))
    #         print(" >> Satellite "+sat+ " Image Summary: Collected: "+str(collectedImageCount)+", Downlinked: "+ str(len(allLatencies))+", latency stats: avg: "+avg+", stdDev: "+std+", max: "+mx+", min: "+mn)
    #     else:
    #         print(" >> Satellite "+sat+ " Image Summary: Collected: "+str(collectedImageCount)+", Downlinked: 0")


    def writeStorageFiles(self):
        dates = ["20260630","20260701","20260702","20260703","20260704","20260705","20260706","20260707","20260708","20260709","20260710"]
        for sat in self.satList:
            timeOffset = 0
            for date in dates:
                self.writeStorageFile(sat, date, timeOffset)
                timeOffset += (3600 * 24)

    def writeStorageFile(self, sat, date, timeOffset):
        self.planExecutionDate = date
        self.readInitialSatStates()
        filenameIn = f"{self.dataPath}planner/output/{date}/{sat}_plan.csv"
        filenameOut = f"{self.dataPath}planner/output/storage/{sat}_storage.csv"
        storage = self.initialSatStates[sat]['storage']
        with open(filenameIn, "r") as fileIn:
            with open(filenameOut, "a") as fileOut:
                for line in fileIn:
                    filteredLine = line.strip()
                    if filteredLine and not filteredLine.startswith("#"):
                        time, cmd = line.split(",")
                        time = int(time.strip()) + timeOffset
                        cmd = cmd.strip()
                        if cmd.lower().startswith("raw"):
                            storage -= (100/60)
                            storage = storage
                        elif cmd.lower().startswith("dnl"):
                            storage += (100/1200)
                            storage = storage
                        fileOut.write(f"{time}, {round(storage,2)}\n")

    def plotStorage(self):
        # Initialize the plot
        plt.figure(figsize=(12, 6))

        # Loop through each satellite to read its file and plot the data
        for sat in self.satList:
            file_name = f"{sat}_storage.csv"
            file_path = os.path.join(f"{self.dataPath}planner/output/storage/", file_name)

            try:
                # Read the CSV file.
                # header=None is used because the sample data doesn't show column headers.
                df = pd.read_csv(file_path, header=None, names=['mission_time', 'available_storage'])

                # Plot the data for the current satellite
                plt.plot(df['mission_time'], df['available_storage'], label=sat)

            except FileNotFoundError:
                print(f"Warning: The file {file_path} was not found.")
            except pd.errors.EmptyDataError:
                print(f"Warning: The file {file_path} is empty.")

        # Formatting the chart
        plt.title("Available Storage % over Mission Duration")
        plt.xlabel("Mission Time (seconds)")
        plt.ylabel("Available Storage (%)")
        plt.legend(title="Satellites", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout so the legend doesn't get cut off
        plt.tight_layout()

        # Display the plot
        plt.show()

    def writeDailyLatencyFiles(self):
        dates = ["20260630","20260701","20260702","20260703","20260704","20260705","20260706","20260707","20260708","20260709","20260710"]
        self.latencySummary = {}
        for date in dates:
            self.latencySummary[date] = []
            self.readLatencyFile(date)
        self.plotLatencySummary()

    def readLatencyFile(self, date):
        filenameIn = f"{self.dataPath}planner/output/{date}/latencies.txt"
        # sat = None
        with (open(filenameIn, "r") as fileIn):
            for line in fileIn:
                cleanLine = line.strip()
                # if cleanLine.lower().startswith("satellite"):
                #     prefix, sat = cleanLine.split(" ")
                #     sat = sat[:-1]
                # elif
                if cleanLine.lower().startswith("{'id': ") and cleanLine.lower().endswith("}"):
                    image = ast.literal_eval(cleanLine)
                    self.latencySummary[date].append(image['latency'])

    def plotLatencySummary(self):
        # Sort the dictionary keys to ensure chronological order
        sorted_date_strings = sorted(self.latencySummary.keys())

        dates = []
        date_labels = []

        # Initialize lists to hold our statistical data
        averages = []
        minimums = []
        maximums = []
        std_devs = []

        # Process dates and calculate stats
        for date_str in sorted_date_strings:
            # Create datetime object for plotting
            dt = datetime.strptime(date_str, '%Y%m%d')
            dates.append(dt)

            # Create the custom "month/date" string without leading zeros
            date_labels.append(f"{dt.month}/{dt.day}")

            # Get data and calculate metrics
            latencies = self.latencySummary[date_str]
            averages.append(np.mean(latencies))
            minimums.append(np.min(latencies))
            maximums.append(np.max(latencies))
            std_devs.append(np.std(latencies))

        # Create the plot
        plt.figure(figsize=(12, 7))

        # Plot the calculated statistics
        plt.plot(dates, averages, label='Average', marker='o', linewidth=2)
        plt.plot(dates, minimums, label='Minimum', marker='v', linestyle='--')
        plt.plot(dates, maximums, label='Maximum', marker='^', linestyle='--')
        plt.plot(dates, std_devs, label='Std Dev', marker='s', linestyle=':')

        # Format the chart
        # plt.title('Latency: Elapsed time between image collection and downlink (minutes)')
        plt.xlabel('Date')
        plt.ylabel('Image latency (minutes)')

        # Apply the custom x-axis ticks and labels
        plt.xticks(dates, date_labels)

        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.6)
        # plt.legend(loc='upper left')
        plt.legend(loc='upper right') #, bbox_to_anchor=(0.02, 0.85))
        plt.tight_layout()
        plt.show()

    # def writeSatLatencyFile(self, sat, date, timeOffset):
    #     self.planExecutionDate = date
    #     filenameIn = f"{self.dataPath}planner/output/{date}/latencies.csv"
    #     filenameOut = f"{self.dataPath}planner/output/{date}/latencySummary.csv"
    #     latencies = {}
    #     with open(filenameIn, "r") as fileIn:
    #         with open(filenameOut, "a") as fileOut:
    #             for line in fileIn:
    #                 filteredLine = line.strip()


                        # time, cmd = line.split(",")
                        # time = int(time.strip()) + timeOffset
                        # cmd = cmd.strip()
                        # if cmd.lower().startswith("raw"):
                        #     storage -= (100/60)
                        #     storage = storage
                        # elif cmd.lower().startswith("dnl"):
                        #     storage += (100/1200)
                        #     storage = storage
                        # fileOut.write(f"{time}, {round(storage,2)}\n")

# DshieldPlanner.writeDailyLatencyFiles()

# DshieldPlanner().plotStorage() #writeStorageFiles()
# DshieldPlanner().writeDailyLatencyFiles()
# DshieldPlanner().writeStorageFiles()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_data_directory', help="filepath to top level directory containing the dshield demo data")
    parser.add_argument('plan_creation_date', help="Date between '20260701' and '20260710' corresponding to a demo date between 7/1/26-7/11/26")
    args = parser.parse_args()
    DshieldPlanner(args.demo_data_directory, args.plan_creation_date).run()


