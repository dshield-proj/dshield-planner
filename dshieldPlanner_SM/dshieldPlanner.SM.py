import ast
import copy
import math
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from gurobipy import quicksum

from solver import Solver
from gp import GP
from collectChartData import DataCollector
import matplotlib.pyplot as plt


class DshieldPlanner:
    def __init__ (self):
        print("DshieldPlanner()")
        self.configFile = "./config/SM.config.json"
        self.config = None
        self.solver = None
        self.satList = None
        self.gsList = None
        self.experimentRoot = None
        self.experimentDate = None
        self.experiment = None
        self.experimentRun = None
        self.targetValFiles = None
        self.angleRange = None
        self.obsRateLarge = None
        self.obsRateSmall = None
        self.dnlRate = None
        self.powerModel = None
        self.maxTick = None
        self.planHorizon = None
        self.cycleDuration = None
        self.cycleIntervals = None
        self.cycleVarIndices = None
        self.maxDownlinkBlockoutSecs = None
        self.minForcedDownlinkSecs = None
        self.includeEnergyConstraints = None
        self.includeSlewFlowConstraints = None
        self.slewEnergyMultiplier = None
        self.includeStorageConstraints = None
        self.includeGsConstraints = None
        self.includeSetupTimeConstraints = None
        self.includeObsOrDnlMutexConstraints = None # required True if DNL and OBS overlap
        self.includeActiveFireTargets = None
        self.includePreFireTargets = None
        self.cmdSetupTime = None
        # self.includeMvars = None # TODO: Remove this experimental variable used to case the sum of downlink secs
        self.rwdThreshold = None
        self.rwdPrecision = None
        self.targetValues = {}
        self.activeFireTargets = {}
        self.unavailableActiveFireTargets = {}
        self.targetTimes = {}
        self.satChoices = {}
        self.eclipses = {}
        self.allSatCycles = {}
        self.allImages = []
        self.allRealImages = []
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
        self.cycleSlewEnergy = None
        self.maxCmdSetupTime = 0

        self.downlinkConflicts = []
        self.satCycleDetails = {}

        self.xVars = None  # x[i] = 1 --> image i in the plan (binary)
        self.yVars = None  # y[j] = 1 --> target j is in the plan (binary)
        self.saVars = None  # sa[s,k] = storage available (%) for sat s on cycle k (0 <= a <= 100)
        self.scVars = None  # sc[s,k] = storage consumed (%) for sat s on cycle k (0 <= a <= 100)
        self.spVars = None  # sp[s,k] = storage produced (%) by s at end of cycle k (0 <= a <= 100)
        self.eaVars = None  # ea[s,k] = energy available (%) for sat s on cycle k (0 <= a <= 100)
        self.eNetVars = None  # eaNet[s,k] = net energy available (%) for sat s on cycle k (0 <= a <= 100)
        self.eauVars = None   # eau[s,k] = 1 <--> 100 <= eNet[s,k-1]
        self.eSlewVars = None   # eSlew[i,j] = 1 <--> scene i is followed by scene j in the plan
        self.sadVars = None  # sad[s,k,n] = storage available at the beginning of dnl window (s,k,n)
        self.scdVars = None  # scd[s,k,n] = storage consumed dnl window (s,k,n)
        self.spdVars = None  # spd[s,k,n] = storage produced during dnl window (s,k,n)

        # old SM planner
        self.gpDict = {}
        self.satEvents = {}
        self.horizonGPs = set()
        self.sortedHorizonGPerr = {}
        self.sortedHorizonGPs = []
        self.sortedGPpct = 0.10
        self.useSortedGP = True
        self.horizonId = 1
        self.horizonDur = 21600 #21600
        self.horizonStart = ((self.horizonId - 1) * self.horizonDur) + 1 #21601 #1 #21601  #1
        self.horizonEnd = self.horizonStart + self.horizonDur - 1
        self.horizonFilter = "noRainOnly" #"noRainOnly"  #"rainOnly" # rainOnly, noRainOnly, all
        self.slewTable = {}
        self.dataPath = None
        self.initialHorizonGpErrAvg = None
        self.errorTable = {}
        self.satEclipses = {}
        self.targetModes = {}
        self._setup_cache = {}

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
        self.readInputs()
        # self.removeZeroValueChoices()
        if self.rwdThreshold and self.rwdThreshold > 0:
            self.removeLowValueChoices()
        self.createDataCycles()
        self.printCycles()
        # self.increaseTargetValues()
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
        self.solver.solveIt()
        self.extractSolution()
        # self.collectSatPlans()
        # self.simulatePlan()
        # self.calculateLatencies()
        self.printCycles()
        # self.writeLatencies()
        # self.writeSelectedTargets()
        # self.writeMissingActiveTargets()
        self.printCycleDetails()

        # self.createResidualRewardHistogram(file="exp3.threshold.0.limit.24hr.txt")
        return
        # self.showSelectedTargetRewardHistogram()
        # self.reportResults()

    def createModel(self):
        print("createModel()")
        self.solver.initSolver(self.config["mipGap"], self.config["timeLimit"])
        self.createVariablesAndObjective()
        self.solver.setObjectiveSense("maximize")
        self.createConstraints()
        # self.solver.writeModel("dshieldFire")

    def createVariablesAndObjective(self):
        print("createVariablesAndObjective()")

        # xVars: x[i] = 1 --> image i is in the plan
        imageVarIndices = [i for i in range(1, len(self.allImages) + 1)]
        self.xVars = self.solver.addBinaryVars(imageVarIndices, "x")

        # for i in self.xVars.keys():
        #     try:
        #         scene = self.getImage(i)
        #     except IndexError:
        #         print(f"\nCRASH DETECTED!")
        #         print(f"Offending key 'i': {i} (Type: {type(i)})")
        #         print(f"Length of allImages: {len(self.allImages)}")
        #         print(f"Value of imageCount: {self.imageCount}")
        #         raise  # Re-raise the error to stop execution
        #
        # print("\n ***** createVariablesAndObjective() NO CRASH *****")


        # Force the START and END dummy nodes to always be in the plan
        for s in self.satList:
            first_cycle = self.allSatCycles[s][0]
            start_id, _ = self.firstAndLastScenesInCycle(first_cycle)
            end_id = self.lastSceneInLastSatCycle(s)
            self.xVars[start_id].lb = 1.0
            self.xVars[end_id].lb = 1.0

        # yVars: y[j,m] = 1 --> target j is selected to be observed with mode m
        sortedTargets = self.sortedHorizonGPs # sorted(self.targetTimes) #
        missingTargets = []
        yVarObjectives = []
        yVarIndices = []
        for target in sortedTargets:
            if target in self.targetModes:
                for mode in self.targetModes[target]:
                    time, sensor1, sensor2 = mode
                    index = (target, time, sensor1, sensor2)
                    yVarIndices.append(index)
                    reward = self.targetModes[target][mode]
                    yVarObjectives.append(reward)
            else:
                print("ERROR! target not found in targetModes: "+str(target))
                missingTargets.append(target)
        self.yVars = self.solver.addBinaryObjectiveVars(yVarIndices, yVarObjectives, "y")

        self.cycleVarIndices = []
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                self.cycleVarIndices.append((s, k))

        if self.includeStorageConstraints:
            # saVars: sa[s,k] available storage on sat s for cycle k
            # self.saVars = self.solver.addContinuousVars(dataVarIndices, "sa", 0, 100)
            o = [1 for index in self.cycleVarIndices]
            # self.saVars = self.solver.addContinuousVars(dataVarIndices, "sa", 0, 100)
            self.saVars = self.solver.addContinuousObjectiveVars(self.cycleVarIndices, o, "sa", 0, 100)

            # TODO: sort out multiple spVars definitions below:
            # spVars: sp[s,k] storage % produced on sat s at end of cycle k
            # self.spVars = self.solver.addContinuousVars(dataVarIndices, "sp", 0, 100)
            # spVar with objective will maximize storage produced each cycle
            self.spVars = self.solver.addContinuousObjectiveVars(self.cycleVarIndices, o, "sp", 0, 500)

            # scVars: sc[s,k] storage % consumed on sat s during cycle k
            self.scVars = self.solver.addContinuousVars(self.cycleVarIndices, "sc", 0, 500)

        if self.includeEnergyConstraints:
            self.createEnergyVariables(self.cycleVarIndices)

        if self.includeGsConstraints:
            self.createGsVariables()

        if self.includeObsOrDnlMutexConstraints:
            self.createDnlVariables()

    def createDnlVariables(self):
        # create variables tracking storage available at the beginning of each downlink window
        indices = []
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = (self.allSatCycles[s][k])
                dnlWindows = cycle["dnl"]
                dnlWindowCount = len(dnlWindows)
                for n in range(dnlWindowCount):
                    dnlWindow = dnlWindows[n]
                    indices.append((s,k,n))
        self.sadVars = self.solver.addContinuousVars(indices, "sad", 0, 100)
        self.scdVars = self.solver.addContinuousVars(indices, "scd", 0, 300)
        self.spdVars = self.solver.addContinuousVars(indices, "spd", 0, 300)

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

        if self.includeSlewFlowConstraints:
            self.createSlewVariables(varIndices)

    def createSlewVariables(self, varIndices):
        MAX_IDLE_GAP_SECONDS = 600  # max allowed idle time between consecutive commands (excluding slew time)
        cmdDuration = 1
        slewVarIndices = []
        lastScenes = {}
        for s in self.satList:
            lastScenes[s] = self.lastSceneInLastSatCycle(s)

        for s, k in varIndices:
            cycle = self.allSatCycles[s][k]
            firstCycleSceneId, lastCycleSceneId = self.firstAndLastScenesInCycle(cycle)
            endSceneId = lastScenes[s]  # The dummy END node for agent s

            # Note: In the first cycle, firstCycleSceneId is the dummy START node.
            for i in range(firstCycleSceneId, lastCycleSceneId + 1):
                # Only process outgoing connections for nodes that are NOT the final END node
                if i != endSceneId:
                    scene_i = self.getImage(i)

                    # INNER LOOP: Find the NEXT REAL SCENES.
                    # Let it look past lastSceneId, all the way to the end of the day (allow slew across cycles)
                    # We stop before endSceneId here, because we handle END manually below
                    for j in range(i + 1, endSceneId):
                        scene_j = self.getImage(j)
                        slewTime = self._get_setup_time(scene_i, scene_j)
                        timeGap = scene_j['time'] - (scene_i['time'] + cmdDuration)

                        # UPPER BOUND: Skip the idle gap check if we are connecting from a dummy node!
                        # Because we separated the END node, we only need to protect START
                        if not self.isStartScene(scene_i):
                            # UPPER BOUND: If j is too far in the future, stop checking this i entirely!
                            # (Because the images are chronologically sorted, everything after j is also too far)
                            # Will naturally stop the loop from going too deep into cycle k + 1
                            if timeGap > slewTime + MAX_IDLE_GAP_SECONDS:
                                break  # Prune future real scenes to save memory!

                            # LOWER BOUND: If j is too soon (physical conflict), skip it, but keep checking the next j
                            if timeGap < slewTime:
                                continue

                        slewVarIndices.append((i, j))  # If we survive both checks, create the variable!

                    # MANUAL END CONNECTION
                    # Guaranteed connection from 'i' to END, regardless of time gaps or breaks!
                    slewVarIndices.append((i, endSceneId))

        # import collections
        # duplicates = [item for item, count in collections.Counter(slewVarIndices).items() if count > 1]
        # assert not duplicates, f"WARNING: Found {len(duplicates)} duplicate transitions! First 5: {duplicates[:5]}"

        self.eSlewVars = self.solver.addBinaryVars(slewVarIndices, "slew")
        print(f"created {len(self.eSlewVars)} slew vars")

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
        startTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"createConstraints() started: {startTime}")

        # if target j is in plan with mode m, then at least one image containing j with mode m is in the plan
        # y[j,m] <= sum(X[i] for all images i containing target j with mode m)                        (1)
        constraintCount = 0
        print(f"c1  start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for yVarIndex in self.yVars:
            target, time, sensor1, sensor2 = yVarIndex
            images = self.imagesWithModeContainingTarget((time, sensor1, sensor2), target)
            self.solver.addConstraint(self.yVars[yVarIndex] <= sum(self.xVars[i] for i in images), f"c1.targetImageInPlan.y[{target},{time},{sensor1},{sensor2}]")
            constraintCount += 1
        print(f"c1 end {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, count: {constraintCount}")

        # (C2) Each target j counts only once in objective regardless of multiple viewings
        # sum of all observation modes for j <= 1
        # sum(y[j,m] forall (j,m)) <= 1:

        constraintCount = 0
        print(f"c2  start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        targetModes = {}
        for yVarIndex in self.yVars:
            target, time, sensor1, sensor2 = yVarIndex
            errorCode = (time, sensor1, sensor2)
            if target not in targetModes:
                targetModes[target] = []
            if errorCode not in targetModes[target]:
                targetModes[target].append(errorCode)

        for target in targetModes:
            targetYvarIndices = []
            for time, sensor1, sensor2 in targetModes[target]:
                index = (target,time, sensor1, sensor2)
                targetYvarIndices.append(index)
            if len(targetYvarIndices) > 1:
                self.solver.addConstraint(sum(self.yVars[index] for index in targetYvarIndices) <= 1, f"c2.targetCountsOnce.{target}")
                constraintCount += 1
        print(f"c2 end {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, count: {constraintCount}")

        constraintCount = 0
        if self.includeStorageConstraints:
            # Storage used on sat s on cycle k = sum of storage used by s on cycle k
            # sc[s,k] = sum( obsRate * x[i] for i in sat s cycle k                          (2)
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = self.allSatCycles[s][k]
                    firstImage, lastImage = self.firstAndLastScenesInCycle(cycle)
                    if firstImage:
                        self.solver.addConstraint(self.scVars[s,k] == sum(self.obsRateLarge * self.xVars[i] for i in range(firstImage, lastImage+1)),
                                                      "c2.storageUsedOnCycle." + s+"."+str(k+1))
                    else:
                        print("createConstraints() No images for sat "+str(s)+", cycle "+str(k))

            # sa[s,0] = 100      Available storage is 100% for first cycle on all sats   (3)
            for s in self.satList:
                self.solver.addConstraint(self.saVars[s,0] == 100, "c3.availableStorageFirstCycle."+s)

            # Available storage for cycle k = available storage for prior cycle - storage used on prior cycle + freed storage at end of prior cycle
            # sa[s,k] = sa[s,k-1] - sc[s,k-1] + sp[s,k-1]                                    (4)
            for s in self.satList:
                for k in range(1,self.cycleCount(s)):
                    self.solver.addConstraint(self.saVars[s,k] == self.saVars[s,k-1] - self.scVars[s,k-1] + self.spVars[s,k-1], "c4.availStorageCycleStart."+s+"."+str(k))

            if self.includeStorageConstraints:
                # used storage must never exceed available storage for any cycle
                # sc[s,k] <= sa[s,k]                                                         (5)
                for s in self.satList:
                    for k in range(self.cycleCount(s)):
                        self.solver.addConstraint(self.scVars[s, k] <= self.saVars[s, k], "c5.neverExceedAvailStorage." + s + "." + str(k))

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
        if self.includeSetupTimeConstraints:
            self.createCommandMutexConstraints()

        if self.includeObsOrDnlMutexConstraints:
            self.createDnlConstraints()

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

    # -----------------------------
    # Setup-time cache
    # -----------------------------
    def _get_setup_time_and_energy(self, typeA, typeB):
        key = (typeA, typeB)
        if key not in self._setup_cache:
            # The slow string parsing only happens once per pair!
            angle1 = int(typeA.split(".")[1])
            angle2 = int(typeB.split(".")[1])
            # Store BOTH in the cache as a tuple
            self._setup_cache[key] = self.getSlewTimeAndEnergy(angle1, angle2)
        return self._setup_cache[key]

    def _get_setup_time(self, cmdA, cmdB):
        if self.isStartScene(cmdA) or self.isEndScene(cmdB):
            return 0
        if "mode" not in cmdA:
            print(f"get_setup_time() error! cmdA: {cmdA}, cmdB: {cmdB}")
        time, energy = self._get_setup_time_and_energy(cmdA["mode"]["type"], cmdB["mode"]["type"])
        return time

        # Add a new method for your new slew energy variables

    def _get_setup_energy(self, cmdA, cmdB):
        if self.isStartScene(cmdA) or self.isEndScene(cmdB):
            return 0
        time, energy = self._get_setup_time_and_energy(cmdA["mode"]["type"], cmdB["mode"]["type"])
        return energy

    def isDummyScene(self, scene):
        return self.isStartScene(scene) or self.isEndScene(scene)

    def isStartScene(self, scene):
        return "startScene" in scene

    def isEndScene(self, scene):
        return "endScene" in scene

    def getStartScene(self, satId):
        cycles = self.allSatCycles[satId]
        startScene = cycles[0]["obs"][0]
        assert self.isStartScene(startScene), f"getStartScene() not a startScene: {startScene}"
        return startScene

    def getEndScene(self, satId):
        cycles = self.allSatCycles[satId]
        endScene = cycles[-1]["obs"][-1]
        assert self.isEndScene(endScene), f"getEndScene() not an endScene: {endScene}"
        return endScene



    # def _get_setup_time(self, typeA, typeB):
    #     key = (typeA, typeB)
    #     if key not in self._setup_cache:
    #         angle1 = int(typeA.split(".")[1])
    #         angle2 = int(typeB.split(".")[1])
    #         time, energy = self.getSlewTimeAndEnergy(angle1, angle2)
    #         self._setup_cache[key] = time
    #     return self._setup_cache[key]
    #
    # def getSlewTimeAndEnergy(self, fromAngle, toAngle):
    #     slewTableRow = self.slewTable[fromAngle]
    #     slewTime, slewEnergy = slewTableRow[toAngle]
    #     slewTimeCeil = math.ceil(slewTime)
    #     return (slewTimeCeil, slewEnergy)

    # -----------------------------
    # Setup-time Conflict predicate
    # -----------------------------
    def isConflict(self, cmdA, cmdB):
        if cmdA["mode"]["type"] == cmdB["mode"]["type"]:
            return False
        dt = cmdB["time"] - cmdA["time"]
        slewSetup = self._get_setup_time(cmdA, cmdB)
        setup = min(slewSetup, 5) #self.maxCmdSetupTime #self._get_setup_time(A["mode"]["type"], B["mode"]["type"])
        return dt < setup

    def createCommandMutexConstraints(self):
        Smax = self.maxCmdSetupTime
        print(f"createCommandMutexConstraints() maxSetupTime {Smax}")
        min_clique_size = 5
        min_neighbors_to_try = 10

        for s in self.satList:
            print(f"sat {s} start {datetime.now()}")

            for k in range(self.cycleCount(s)):
                pairwiseCount = 0
                cliqueCount = 0

                cycle = self.allSatCycles[s][k]
                firstImage, lastImage = self.firstAndLastScenesInCycle(cycle)

                if firstImage:
                    # ----------------------------------
                    # Step 1: Build conflict graph
                    # ----------------------------------
                    conflicts_map = {
                        i: set() for i in range(firstImage, lastImage + 1) if not (self.isDummyScene(self.getImage(i)))
                    }

                    for i in range(firstImage, lastImage + 1):
                        A = self.getImage(i)
                        if not self.isDummyScene(A):
                            t1 = A["time"]
                            type1 = A["mode"]["type"]

                            for j in range(i + 1, lastImage + 1):
                                B = self.getImage(j)
                                if not self.isDummyScene(B):
                                    dt = B["time"] - t1
                                    if dt >= Smax:
                                        break

                                    if type1 != B["mode"]["type"]:
                                        if self.isConflict(A, B):
                                            conflicts_map[i].add(j)
                                            conflicts_map[j].add(i)

                    # ----------------------------------
                    # Step 2: Build constraints
                    # ----------------------------------
                    for i in conflicts_map.keys():

                        neighbors = list(conflicts_map[i])

                        # ---- greedy clique using graph
                        clique = []
                        if len(neighbors) >= min_neighbors_to_try:
                            for v in neighbors:
                                if all(u in conflicts_map[v] for u in clique):
                                    clique.append(v)

                        full_clique = [i] + clique
                        clique_set = set(clique)

                        # ---- clique constraint
                        if len(full_clique) >= min_clique_size:
                            self.solver.addConstraint(
                                quicksum(self.xVars[x] for x in full_clique) <= 1,
                                f"cx.cmdClique.{s}.{k}.{i}"
                            )
                            cliqueCount += 1

                        # ---- pairwise constraints
                        for j in neighbors:
                            if j in clique_set:
                                continue

                            if i < j:  # avoid duplicates
                                self.solver.addConstraint(
                                    self.xVars[i] + self.xVars[j] <= 1,
                                    f"cx.cmdMutex.{s}.{k}.{i}.{j}"
                                )
                                pairwiseCount += 1

                    print(
                        f"sat {s}, cycle {k}, pairwise: {pairwiseCount}, "
                        f"cliques: {cliqueCount}, end {datetime.now()}"
                    )

                    # ---- Calculate Density safely inside the block!
                    total_edges = sum(len(v) for v in conflicts_map.values())
                    n = len(conflicts_map.keys())  # Safer than max-min, accounts for skipped dummy nodes!
                    if n > 1:
                        density = total_edges / (n * (n - 1) / 2)
                        print(f"setup edge density for sat {s}, cycle {k}: {density:.3f}")

    def createDnlConstraints(self):
        # ensure any second when obs and dnls overlap is consumed by either obs or dnl but not both
        # subtract # of selected observations from # of dnl seconds in overlapping dnl window
        # p[s,k,g,n] <= (duration of n) - sum(images selected in k before n)
        for s in self.satList:
            for k in range(self.cycleCount(s)):
                cycle = self.allSatCycles[s][k]
                obs = cycle["obs"] if "obs" in cycle else None
                dnlWindows = cycle["dnl"] if "dnl" in cycle else None
                if obs and dnlWindows:
                    dnlWindowIndex = -1
                    for dnlWindow in dnlWindows:
                        dnlWindowIndex += 1
                        start = dnlWindow["start"]
                        end = dnlWindow["end"]
                        g = dnlWindow["gs"]
                        obsVars = []
                        for o in obs:
                            if start <= o["time"] and o["time"] <= end:
                                var = self.xVars[o["image"]]
                                obsVars.append(var)
                        if obsVars:
                            pVar = self.pVars[(s,k,g,dnlWindowIndex)]
                            dur = self.dnlSlotDuration(s,k,dnlWindowIndex)
                            self.solver.addConstraint(pVar <= dur - sum(obsVars), "c30."+s+"."+str(k)+"."+str(dnlWindowIndex))

            # sad[s,k,n] = sa[s,k] - (storage consumed in cycle k before n) + (storage produced in cycle k before n)
            indices = []
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = (self.allSatCycles[s][k])
                    dnlWindows = cycle["dnl"] if "dnl" in cycle else []
                    for n in range(len(dnlWindows)):
                        xVars = self.collectCycleXvarsBeforeDnlWindow(cycle, dnlWindows[n])
                        pVars = self.collectCyclePvarsBeforeDnlWindow(s,k,n, dnlWindows)
                        saVar = self.saVars[(s,k)]
                        sadVar = self.sadVars[(s,k,n)]
                        #TODO: fix bug here where it assumes all images are large images
                        # Should do something like this:
                        # sum(self.obsRateLarge * self.xVars[i] for i in largeImages) + sum(self.obsRateSmall * self.xVars[j] for j in smallImages)
                        self.solver.addConstraint(sadVar == saVar - (sum(xVars) * self.obsRateLarge) + (sum(pVars) * self.dnlRate), "c31."+str(s)+"."+str(k)+"."+str(n))


            # scd[s, k, n] = # of selected images * rate
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = (self.allSatCycles[s][k])
                    dnlWindows = cycle["dnl"]
                    if dnlWindows:
                        for n in range(len(dnlWindows)):
                            dnlWindow = dnlWindows[n]
                            xVars = self.collectCycleXvarsInDnlWindow(cycle, dnlWindow)
                            scdVar = self.scdVars[(s,k,n)]
                            self.solver.addConstraint(scdVar == sum(xVars) * self.obsRateLarge, "c32.scd["+str(s)+","+str(k)+","+str(n)+"]")


            # spd[s, k, n] = # of downlink seconds planned * rate
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = (self.allSatCycles[s][k])
                    dnlWindows = cycle["dnl"]
                    if dnlWindows:
                        for n in range(len(dnlWindows)):
                            dnlWindow = dnlWindows[n]
                            g = dnlWindow["gs"]
                            spdVar = self.spdVars[(s,k,n)]
                            pVar = self.pVars[(s,k,g,n)]
                            self.solver.addConstraint(spdVar == pVar * self.dnlRate, "c33.spd["+str(s)+","+str(k)+","+str(n)+"]")

            # spd[s, k, n] <= 100 - (sad[s,k,n] - scd[s,k,n])
            for s in self.satList:
                for k in range(self.cycleCount(s)):
                    cycle = (self.allSatCycles[s][k])
                    dnlWindows = cycle["dnl"]
                    if dnlWindows:
                        for n in range(len(dnlWindows)):
                            index = (s,k,n)
                            spdVar = self.spdVars[index]
                            sadVar = self.sadVars[index]
                            scdVar = self.scdVars[index]
                            # TODO: put this constraint back in!!!!
                            self.solver.addConstraint(scdVar <= sadVar + spdVar, "c34.scd["+str(s)+","+str(k)+","+str(n)+"].cap")


    def collectCycleXvarsBeforeDnlWindow(self, cycle, dnlWindow):
        #TODO: ensure dnlWindows are sorted by start time
        obsVars = []
        for image in cycle["obs"]:
            if image["time"] < dnlWindow["start"]:
                var = self.xVars[image["image"]]
                obsVars.append(var)
        return obsVars

    def collectCycleXvarsInDnlWindow(self, cycle, dnlWindow):
        #TODO: ensure dnlWindows are sorted by start time
        obsVars = []
        for image in cycle["obs"]:
            if dnlWindow["start"] <= image["time"] and  image["time"] <= dnlWindow["end"]:
                var = self.xVars[image["image"]]
                obsVars.append(var)
        return obsVars

    def collectCyclePvarsBeforeDnlWindow(self, s,k,n, dnlWindows):
        #TODO: ensure dnlWindows are sorted by start time
        pVars = []
        for i in range(n):
            dnlWindow = dnlWindows[i]
            g = dnlWindow["gs"]
            pVars.append(self.pVars[(s,k,g,i)])
        return pVars



    def createEnergyConstraints(self):
        print("createEnergyConstraints()")
        # ea[s,0] = 100      Available energy is 100% for first cycle on all sats   (8)
        for s in self.satList:
            self.solver.addConstraint(self.eaVars[s, 0] == 100, "c8.availableEnergyFirstCycle." + s)
        if self.includeSlewFlowConstraints:
            self.createSlewFlowConstraints()
            self.createSlewEnergyConstraints(self.cycleVarIndices)

        # # never dip below minimum energy threshold
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
                cycle["eclipseDur"] = cycleEclipseDur #self.getEclipseTickCount(s, cycle["start"], cycle["end"])
                pList = []
                for n in range(len(cycle["dnl"])):
                    g = cycle["dnl"][n]["gs"]
                    pList.append(self.pVars[(s, k, g, n)])
                self.solver.addConstraint(self.eNetVars[s, k] == self.eaVars[s, k] + powerIn - powerOutDefault
                                          - (powerOutDnl * sum(pList))- self.cycleSlewEnergy[s,k], "c10.netEnergy." + s + "." + str(k))

        # ea[s,k] = min(eNet[s,k-1], 100)  energy is capped at 100 % despite constant solar panel exposure    (12)
        self.createMaxEnergyConstraints()



        # self.createActiveFireConstraints()

    # def createActiveFireConstraints(self):
    #     print("createActiveFireConstraints()")
    #     for target in self.activeFireTargets:
    #         if target in self.yVars:
    #             yVar = self.yVars[target]
    #             self.solver.addConstraint(yVar == 1, "c.30.activeFire."+str(target))


    def createMaxEnergyConstraints(self):
        print(f"createMaxEnergyeConstraints()")
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

    def createSlewFlowConstraints(self):
        print(f"createSlewFlowConstraints() slewVar count {len(self.eSlewVars)}")

        # 1. Fast lookup dictionaries for edges
        flow_out = defaultdict(list)
        flow_in = defaultdict(list)

        for (i, j), var in self.eSlewVars.items():
            flow_out[i].append(var)
            flow_in[j].append(var)

        out_count = 0
        in_count = 0

        # Loop over ALL variables to guarantee balance
        for i in self.xVars.keys():
            scene = self.getImage(i)

            # Flow-Out constraint (Everyone gets this EXCEPT the END node)
            if not self.isEndScene(scene):
                self.solver.addConstraint(
                    quicksum(flow_out[i]) == self.xVars[i],
                    f"c.15.flowOut.{i}"
                )
                out_count += 1

            # Flow-In constraint (Everyone gets this EXCEPT the START node)
            if not self.isStartScene(scene):
                self.solver.addConstraint(
                    quicksum(flow_in[i]) == self.xVars[i],
                    f"c.16.flowIn.{i}"
                )
                in_count += 1

        print(f"Created {out_count} Flow-Out and {in_count} Flow-In constraints.")

    def createSlewEnergyConstraints(self, varIndices):
        # Create the intermediate continuous variables
        self.cycleSlewEnergy = self.solver.addContinuousVars(varIndices, "cycleSlewEnergy", 0, 100)

        # 1. Initialize an empty list for every cycle
        cycle_slew_exprs = {(s, k): [] for s, k in varIndices}

        # 2. Loop over the EXACT valid variables we already created
        for (i, j), slew_var in self.eSlewVars.items():
            scene_i = self.getImage(i)
            scene_j = self.getImage(j)

            # Find out which cycle 'i' belongs to
            s = scene_i['sat']
            k = scene_i['cycle']

            # Get the cost and add it to that cycle's bucket
            energy_cost = self._get_setup_energy(scene_i, scene_j) * self.slewEnergyMultiplier
            cycle_slew_exprs[(s, k)].append(slew_var * energy_cost)

        # 3. Define the intermediate variables
        for s, k in varIndices:
            self.solver.addConstraint(
                self.cycleSlewEnergy[s, k] == quicksum(cycle_slew_exprs[(s, k)]),
                f"cx.cycleSlewEnergy.{s}.{k}"
            )

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

    def getCycleStartTime(self, cycle):
        # called only by addCycleStartAndEndTimes()
        firstDnl = None
        firstObs = None
        if cycle["dnl"]:
            firstDnlWindow = cycle["dnl"][0]
            firstDnl = firstDnlWindow["start"]
        if cycle["obs"]:
            firstObservation = cycle["obs"][0]
            firstObs = firstObservation["time"]
        if firstDnl and firstObs:
            if firstObs < firstDnl:
                return firstObs
            else:
                return firstDnl
        elif firstDnl:
            return firstDnl
        elif firstObs:
            return firstObs
        else:
            return 0

    def getCycleEndTime(self, cycle):
        # called only by addCycleStartAndEndTimes()
        lastDnl = None
        lastObs = None
        if cycle["dnl"]:
            lastDnlWindow = cycle["dnl"][-1]
            lastDnl = lastDnlWindow["end"]
        if cycle["obs"]:
            lastObservation = cycle["obs"][-1]
            lastObs = lastObservation["time"]
        if lastDnl and lastObs:
            if lastObs > lastDnl:
                return lastObs
            else:
                return lastDnl
        elif lastDnl:
            return lastDnl
        elif lastObs:
            return lastObs
        else:
            return self.planHorizon


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
        # self.validateSolution(self.selectedTargets, self.selectedImages)

    def updateCyclesWithPlan(self):
        for i in self.selectedImages:
            image = self.getImage(i)
            sat = image["sat"]
            k = image["cycle"]
            # if k in self.allSatCycles[sat]:
            satCycle = self.allSatCycles[sat][k]
            if "selectedImages" not in satCycle:
                satCycle["selectedImages"] = []
            satCycle["selectedImages"].append(i)
            # else:
            #     print("updateCyclesWithPlan() missing cycle! sat: "+str(sat)+", cycle: "+str(k))
        if self.includeStorageConstraints:
            for sat in self.satList:
                for k in range(self.cycleCount(sat)):
                    satCycle = self.allSatCycles[sat][k]
                    if "availSpace" not in satCycle:
                        satCycle["availSpace"] = None
                    if "usedSpace" not in satCycle:
                        satCycle["usedSpace"] = None
                    if "freedSpace" not in satCycle:
                        satCycle["freedSpace"] = None
                    satCycle["availSpace"]  = self.solver.getVarValue(self.saVars[sat,k])
                    satCycle["usedSpace"]   = self.solver.getVarValue(self.scVars[sat,k])
                    satCycle["freedSpace"]  = self.solver.getVarValue(self.spVars[sat,k])
                    satCycle["dnlPlanSecs"] = round(satCycle["freedSpace"]/self.dnlRate,3)

                    if self.includeGsConstraints:
                        satCycle["dnlPlan"] = self.collectCycleDnlPlan(sat,k)
                    if self.includeEnergyConstraints:
                        satCycle["energyAvail"] = self.solver.getVarValue(self.eaVars[sat,k])
                        # satCycle["energyRaw"] = self.solver.getVarValue(self.eaRawVars[sat,k])
                        satCycle["energyNet"] = self.solver.getVarValue(self.eNetVars[sat,k])

    def collectCycleDnlPlan(self, sat, cycle):
        plan = {"z":[],"t":[],"p":[],"sad": [],"scd": [],"spd": [], "tNoP":[], "u":[], "w":[], "v": []}
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
                    index = (s,k,n)
                    if index in self.sadVars:
                        val = round(self.solver.getVarValue(self.sadVars[index]),3)
                        if val > 0.0:
                            plan["sad"].append((n, val))
                        val = round(self.solver.getVarValue(self.scdVars[index]), 3)
                        if val > 0.0:
                            plan["scd"].append((n, val))
                        val = round(self.solver.getVarValue(self.spdVars[index]), 3)
                        if val > 0.0:
                            plan["spd"].append((n, val))
        plan["sad"] = sorted(plan["sad"])
        plan["scd"] = sorted(plan["scd"])
        plan["spd"] = sorted(plan["spd"])
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
            imageDict = self.allRealImages[image-1]
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
            image1 = self.allRealImages[imageId1-1]
            sat1 = image1["sat"]
            for imageId2 in selectedImages:
                if imageId2 > imageId1:
                    image2 = self.allRealImages[imageId2-1]
                    sat2 = image2["sat"]
                    if sat1 == sat2:
                        if image2["time"] - image1["time"] < self.cmdSetupTime: # TODO:  and image1["type"] != image2["type"]:
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
            image = self.allRealImages[imageId-1]
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
                obsDataRate = self.obsRateLarge # if image["type"] == "large" else self.obsRateSmall
                satState["availSpace"] -= obsDataRate
                # if satState["availSpace"] <= 0:
                #     print("simulatePlan() ERROR! storage below empty! sat: "+satId+", image: "+str(image)+", satState: "+str(satState))
                # self.plans[satId].append({"image": image, "state": satState})

            # TODO: simulate downlinks!


    def calculateLatencies(self):
        print("\nCalculating Latencies")
        for sat in self.allSatCycles:
            self.calculateSatLatencies(sat)

    def collectSatObsPlanForCycle(self, sat, cycle):
        cycleImages = []
        for obs in self.satPlans[sat]:
            if obs["cycle"] == cycle:
                cycleImages.append(self.getImage(obs["image"]))
        return cycleImages

    def calculateSatLatencies(self, sat):
        print("\nSatellite: "+sat)
        allLatencies = []
        buffer = []
        cycleId = -1
        collectedImageCount = 0
        for cycle in self.allSatCycles[sat]:
            cycleCollectedImageCount = len(cycle["selectedImages"]) if "selectedImages" in cycle else 0
            collectedImageCount += cycleCollectedImageCount
            cycleId +=1
            cycleImages = self.collectSatObsPlanForCycle(sat, cycleId)
            bufferSize = len(buffer)
            totalImages = bufferSize + cycleCollectedImageCount
            print("\n============\n\nSatellite: "+sat+", Cycle "+str(cycleId)+": "+ str(bufferSize)+" images in buffer + " +str(cycleCollectedImageCount)+ " images collected = "+str(totalImages))
            print("| Image | Collected | Downlinked | Latency | Remaining Images |")
            if "selectedImages" in cycle:
                for imageId in cycle["selectedImages"]:
                    time = self.getImage(imageId)["time"]
                    buffer.append(
                        {"id": imageId, "collectionTime": time, "remainingPct": 100})
            if buffer:
                dnlPlan = cycle["dnlPlan"] if "dnlPlan" in cycle else None
                if dnlPlan:
                    currentImage = buffer[0]
                    # print("Current image: sat " + str(sat) + ", cycle " + str(cycleId) + ", " + str(
                    #     currentImage)+", imageCount: "+str(len(buffer)))
                    currentImageCollectionTime = currentImage["collectionTime"]
                    for dnlWindowIndex in range(len(cycle["dnl"])): #dnlPlanTimes)):
                        dWindow = cycle["dnl"][dnlWindowIndex]
                        if buffer and currentImage:
                            dnlStartTime = dWindow["start"] #int(dnlPlanTimes[dnlWindowIndex][1])
                            dnlEndTime = dWindow["end"]
                            if currentImageCollectionTime < dnlEndTime:
                                for tick in range(dnlStartTime, dnlEndTime+1):
                                    # subtract 5 % image remaining per second (20 seconds to downlink an image)
                                    if currentImageCollectionTime <= tick:
                                        currentImage["remainingPct"] = round(currentImage["remainingPct"] - 5, 3)
                                        if currentImage["remainingPct"] <=0:
                                            currentImage["downlinkTime"] = tick
                                            latency = round((currentImage["downlinkTime"] - currentImage["collectionTime"] + 1)/60, 2)
                                            assert latency > 0, "calcualteSatLatencies() ERROR! Negative Latency: sat: "+str(sat)+", cycle: "+str(cycle)+", image: "+str(currentImage)+", latency: "+str(latency)
                                            currentImage["latency"] = latency
                                            allLatencies.append(currentImage["latency"])
                                            if "latencies" not in cycle:
                                                cycle["latencies"] = []
                                            downlinkedImage = buffer.pop(0)
                                            cycle["latencies"].append(downlinkedImage)
                                            imageInfo = str(downlinkedImage["id"])+", collected: "+str(downlinkedImage["collectionTime"])+", downlinked: "+str(downlinkedImage["downlinkTime"])+", latency: "+str(downlinkedImage["latency"])
                                            imageTableInfo = "| " +str(downlinkedImage["id"]).rjust(5," ")+" | "+str(downlinkedImage["collectionTime"]).rjust(9, " ")+" | "+str(downlinkedImage["downlinkTime"]).rjust(10, " ")+" | "+str(downlinkedImage["latency"]).rjust(7, " ")+" | "+str(len(buffer)).rjust(8," ")+"        |"
                                            print(imageTableInfo)
                                            if buffer:
                                                currentImage = buffer[0]
                                                currentImageCollectionTime = currentImage["collectionTime"]
                                            else:
                                                currentImage = None
                                                print("Buffer Empty! sat " + str(sat) + ", cycle " + str(cycleId)+"\n")
                                                break
            # end for cycle
        if allLatencies:
            avg = str(round(np.average(allLatencies),2))
            std = str(round(np.std(allLatencies),2))
            mx = str(round(np.max(allLatencies),2))
            mn = str(round(np.min(allLatencies),2))
            print(" >> Satellite "+sat+ " Image Summary: Collected: "+str(collectedImageCount)+", Downlinked: "+ str(len(allLatencies))+", latency stats: avg: "+avg+", stdDev: "+std+", max: "+mx+", min: "+mn)
        else:
            print(" >> Satellite "+sat+ " Image Summary: Collected: "+str(collectedImageCount)+", Downlinked: 0")


    def getImage(self, imageId):
        return self.allImages[imageId-1]

    def cycleCount(self, sat):
        return len(self.allSatCycles[sat])


    #####  Read Input Data ######

    def readInputs(self):
        print("readInputs()")
        self.config = self.readConfigFile()
        print("config: "+str(self.config))
        self.setConfigParams()
        self.loadPreprocessingResults()  # sorts and cuts targets down to top sortedGPpct
        self.initPowerModel()
        self.collectTargetModesAndRewards()
        # self.readTargetValueFiles() # TODO: Fix this (for reading Kurtis' pre-fire targets)
        # self.readActiveFireTargets()
        # self.increaseTargetValues()
        for sat in self.satList:
            self.readSatChoiceFile(sat)
            choices = list(self.satChoices[sat].keys())
            print("tp count for " + sat + ": " + str(len(choices)) + ", TP range: " + str(choices[0]) + " - " + str(
                choices[-1]))
            if self.config["useCase"] == "SM":
                self.combineEclipseFilesForSatSM(sat)
            else:
                self.readEclipseFileForSat(sat)
        print("target modes index count (target count): " + str(len(self.targetModes.keys()))) # TODO: why is this still 764197 vs. 76419?


    def readConfigFile(self):
        with open(self.configFile, 'r') as f:
            msg = ""
            lines = f.readlines()
            for line in lines:
                l = line.strip()
                # strip out comments
                pos = l.find("#")
                if pos >= 0:
                    l = l[:pos]
                msg += l
            config = eval(msg)
        return config

    def setConfigParams(self):
        self.solver = Solver(self.config["solver"])
        self.satList = self.config["satList"]
        self.gsList = self.config["gsList"]
        self.dataPath = self.config["dataPath"]
        self.experimentRoot = self.config["experimentRoot"]
        self.experimentDate = self.config["experimentDate"]
        self.experiment = self.experimentRoot+"/"+self.experimentDate
        self.experimentRun = self.config["experimentRun"]
        self.targetValFiles = self.config["targetValFiles"]
        self.obsRateLarge = self.config["obsRateLarge"]
        self.obsRateSmall = self.config["obsRateSmall"]
        self.dnlRate = self.config["dnlRate"]
        self.powerModel = self.config["powerModel"]
        self.planHorizon = self.config["planHorizon"]
        self.angleRange = self.config["angleRange"]
        self.maxTick = self.config["maxTick"]
        self.includeEnergyConstraints = self.config["includeEnergyConstraints"]
        self.includeSlewFlowConstraints = self.config["includeSlewFlowConstraints"]
        self.slewEnergyMultiplier = self.config["slewEnergyMultiplier"]
        self.includeStorageConstraints = self.config["includeStorageConstraints"]
        self.includeGsConstraints = self.config["includeGsConstraints"]
        self.includeSetupTimeConstraints = self.config["includeSetupTimeConstraints"]
        self.includeObsOrDnlMutexConstraints = self.config["includeObsOrDnlMutexConstraints"]
        self.includeOnlyDnlOverlappingWithObs = self.config["includeOnlyDnlOverlappingWithObs"]
        self.includePreFireTargets = self.config["includePreFireTargets"]
        self.includeActiveFireTargets = self.config["includeActiveFireTargets"]
        self.cmdSetupTime = self.config["cmdSetupTime"]
        # self.includeMvars = self.config["includeMvars"]
        self.rwdThreshold = self.config["rwdThreshold"]
        self.rwdPrecision = self.config["rwdPrecision"]
        self.cycleDuration = self.config["cycleDuration"]

    def readSatChoiceFile(self, sat):
        satChoices = {}  # {TP: {sourceID: [gpList]}}
        filepath = self.dataPath +f"sat{sat}/s{sat}.choices.txt"   # ./inputs/" + self.experiment + "/planner/" + self.experimentRun + "/"
        # filepath = "./inputs/" + self.experiment + "/planner/" + self.experimentRun + "/"
        # filenames = os.listdir(filepath)
        # filename = None
        # for file in filenames:
        #     if file.startswith(sat + "_choices"):
        #         filename = file
        #         break
        # filepath += filename

        print("readSatChoiceFile() reading file for " + sat + ": " + filepath)
        with open(filepath, "r") as f:
            for line in f:
                # line format: "32498: [{'cmd': 'obs', 'targets': [11928, 11929]}, {'cmd': 'DNL', 'targets': ['MerrittIsland']}]"
                filteredLine = line.strip()
                if filteredLine and not filteredLine.startswith("--- GAP"):
                    dict = "{" + filteredLine + "}"
                    choices = ast.literal_eval(dict)
                    tp = list(choices.keys())[0]
                    if self.planHorizon and tp > self.planHorizon:
                        break
                    if len(choices) > 1:
                        print("multiple cmd choices: "+str(choices))
                    satChoices.update(choices)
                    # print("choices: "+str(choices))
        self.satChoices[sat] = satChoices

    def collectTargetModesAndRewards(self):
        for gpID in self.sortedHorizonGPs: #.gpDict:
            gp = self.gpDict[gpID]
            self.targetModes[gpID] = {}
            modes = {}
            for choice in gp.errorTableChoices:
                if choice["obs"] == 1:
                    gpDictKeys = list(choice.keys())
                    tick = None
                    for k in gpDictKeys:
                        if isinstance(k, int):
                            tick = k
                            break
                    modelTime = self.getModelTime(tick)
                    initialError = None
                    for e in gp.initialModelError:
                        if e[0] == modelTime:
                            initialError = e[1]
                    sensor1, sensor2 =  choice["row"]
                    errorValue = choice["err"]
                    reward = initialError - errorValue
                    mode = (modelTime, sensor1, sensor2)
                    if mode not in modes and reward > 0:
                        modes[mode] = reward
                    # else:
                    #     assert modes[mode] == reward or reward <= 0, "collectTargetModesAndRewards() ERROR! conflicting mode "+str(mode)+" and reward: "+str(reward)+" vs. "+str(modes[mode])
            self.targetModes[gpID] = modes



    def readTargetValueFiles(self):
        # TODO: Fix This
        return
        filepath = "./inputs/"+self.experiment+"/target_value/" + self.experimentRun + "/"
        # filepath = "./inputs/" + self.experiment + "/planner/" + self.experimentRun + "/"
        # filepath = "./inputs/"+self.experiment+"/"
        filenames = os.listdir(filepath)
        for targetFile in self.targetValFiles:
            fileFound = False
            for file in filenames:
                if file == targetFile:
                    self.readTargetValues(filepath + targetFile)
                    fileFound = True
            assert fileFound, "readTargetValues() ERROR! file not found: "+targetFile

    def readTargetValues(self, filepath):
        print("readTargetValues() reading file: " + filepath)
        with open(filepath, "r") as f:
            firstLine = True
            for line in f:
                if firstLine:
                    firstLine = False
                    continue
                filteredLine = line.strip()
                if filteredLine:
                    gp, value = filteredLine.split(",")
                    gp = int(gp)
                    value = float(value)
                    if self.rwdPrecision:
                        value = round(value,self.rwdPrecision)
                    if gp not in self.targetValues or value > self.targetValues[gp]:
                        self.targetValues[gp] = value

    def readActiveFireTargets(self):
        filepath = "./inputs/"+self.experiment+"/target_value/" + self.experimentRun + "/TV_ACTIVE_"+str(self.experimentDate[:4])+".csv"
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                row = line.strip().split(",")
                target = int(row[0])
                val = float(row[3])
                if target in self.targetTimes:
                    self.activeFireTargets[target] = val
                else:
                    self.unavailableActiveFireTargets[target] = val
        print("readActiveValueTarget() count: "+str(len(self.activeFireTargets)))
        print("readActiveValueTarget() unavailable ("+str(len(self.unavailableActiveFireTargets))+"):")
        print(str(self.unavailableActiveFireTargets))

    def increaseTargetValues(self):
        minActiveTargetRwd = 50000
        maxPreTargetRwd = 0
        minActiveTargetRwdAdjusted = 50000
        maxPreTargetRwdAdjusted = 0
        availableRewards = 0
        for target in self.targetValues:
            if target in self.targetTimes:
                if target in self.activeFireTargets:
                    if self.targetValues[target] < minActiveTargetRwd:
                        minActiveTargetRwd = self.targetValues[target]
                    self.targetValues[target] = 1000 * self.targetValues[target]
                    availableRewards += self.targetValues[target]
                    if self.targetValues[target] < minActiveTargetRwdAdjusted:
                        minActiveTargetRwdAdjusted = self.targetValues[target]
                else:
                    if self.targetValues[target] > maxPreTargetRwd:
                        maxPreTargetRwd = self.targetValues[target]
                    self.targetValues[target] = 100 * self.targetValues[target]
                    availableRewards += self.targetValues[target]
                    if self.targetValues[target] > maxPreTargetRwdAdjusted:
                        maxPreTargetRwdAdjusted = self.targetValues[target]
        print("\nMin active target rwd: "+str(minActiveTargetRwd)+", max pre target rwd: "+str(maxPreTargetRwd)+"\n")
        print("\nMin active target rwd (adjusted): "+str(minActiveTargetRwdAdjusted)+", max pre target rwd (adjusted: "+str(maxPreTargetRwdAdjusted)+"\n")
        print("\nTotal Available Rewards: "+str(round(availableRewards,3)))


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


    def findSelectedActiveFireTargets(self):
        experimentDir = "./IJCAI_SocialGood_Results/20240226/ActiveAndPre/threshold.0.005/"
        filepathSelected = experimentDir + "selectedTargets.txt"
        selectedTargets = []
        activeTargets = []
        with open(filepathSelected, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("["):
                    selectedTargets = ast.literal_eval(line)
                    break
        filepathActive = "./IJCAI_SocialGood_Results/20240226/"
        filepathActive += "activeFireTargets.20240226.txt"
        with open(filepathActive, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("["):
                    activeTargets = ast.literal_eval(line)
                    break
        missingActiveTargets = []
        missingRewards = 0
        for active in activeTargets:
            if active not in selectedTargets:
                missingActiveTargets.append(active)
                missingRewards += self.targetValues[active]
        with open (experimentDir+"missingActiveTargets.txt", "w") as f:
            msg = "# Missing Active Fire Targets " + self.experimentDate + " (" + str(len(missingActiveTargets)) + "),  missed rewards total: "+str(missingRewards)
            print(msg)
            f.write(msg)
            msg = "\n" + str(missingActiveTargets)
            print(msg)
            f.write("\n"+msg)

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



    def calculateCycleIntervals(self):
        self.cycleIntervals = []
        start = 0
        if self.planHorizon < self.cycleDuration:
            end = self.planHorizon -1
            self.cycleIntervals.append({"start": start,"end": end})
        else:
            end = start + self.cycleDuration - 1
            while end < self.planHorizon:
                self.cycleIntervals.append({"start": start, "end": end})
                if end == self.planHorizon - 1:
                    break
                start = end + 1
                end = start + self.cycleDuration - 1
                if end > self.planHorizon:
                    end = self.planHorizon - 1

    # def calculateCycleIntervals(self):
    #     self.cycleIntervals = []
    #     start = 0
    #     end = start + self.cycleDuration-1
    #     while end < self.planHorizon:
    #         self.cycleIntervals.append({"start": start,"end": end})
    #         if end == self.planHorizon -1:
    #             break
    #         start = end+1
    #         end = start + self.cycleDuration - 1
    #         if end > self.planHorizon:
    #             end = self.planHorizon-1

    def getFirstObsTick(self, timepoints, cmdChoices):
        for tp in timepoints:
            choices = cmdChoices[tp]
            for cmd in choices:
                if self.isObsCmd(cmd):
                    return tp
        return None

    def getCurrentCycleStopTime(self, tp):
        for i in self.cycleIntervals:
            if i["start"] <= tp and tp <= i["end"]:
                return i["end"]
        return None

    def getCurrentCycle(self, tp, cycles):
        i = 0
        for c in cycles:
            if c["start"] <= tp and tp <= c["end"]:
                return c, i
            i += 1
        return None, -1

    def createDataCycles(self):
        self.imageCount = 1
        self.calculateCycleIntervals()
        removedImages = []
        for satId in self.satChoices:
            satImageRange = {"firstImage": self.imageCount}

            print(f"createDataCycles() sat {satId}, start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # commandChoices are the rows in the choice file [ {"cmd": x, "targets": yz}, ...}
            # Ex:  [{'cmd': 'obs', 'targets': [24398, 24455]}, {'cmd': 'obs-S', 'targets': [24456]}, {'cmd': 'DNL', 'targets': {'MI'}}]
            cmdChoices = self.satChoices[satId]
            timepoints = sorted(cmdChoices.keys())
            firstObsTick = self.getFirstObsTick(timepoints, cmdChoices)
            # skip downlinks until the first observation
            filteredTimepoints = [tp for tp in timepoints if tp >= firstObsTick]

            cycles = []
            for i in self.cycleIntervals:
                cycle = {"start": i["start"], "end": i["end"], "obs": [], "dnl": []}
                cycles.append(cycle)
            openDnlWindows = {}
            # currentCycleStopTime = self.getCurrentCycleStopTime(filteredTimepoints[0])
            priorCycle = None

            # create dummy start scene
            startScene = {"sat": satId, "time": -1 , "cycle": 0, "image": self.imageCount, "startScene": True }
            cycles[0]["obs"].append(startScene)
            self.allImages.append(startScene)
            self.imageCount += 1
            print(f"createDataCycles() sat {satId} startScene: {startScene}")

            for tp in filteredTimepoints:
                modelTime = self.getModelTime(tp)
                cycle, cycleIndex = self.getCurrentCycle(tp, cycles)
                if priorCycle and priorCycle != cycle:
                    openDnlWindows = {}
                priorCycle = cycle

                choices = cmdChoices[tp]
                if cycle:
                    for cmd in choices:
                        if self.isObsCmd(cmd):
                            targets = self.getObsTargets(cmd)
                            mode = cmd["cmd"]
                            if self.isValidCmdAngle(mode):
                                sensor1, sensor2 = self.getErrorTableCode(mode)
                                errorMode = (modelTime, sensor1, sensor2)
                                # Remove low-value targets (sortedGpPct) and remove images where all targets are removed
                                filteredTargets = [t for t in targets if t in self.sortedHorizonGPs]
                                if filteredTargets:
                                    mode = cmd["cmd"]
                                    sensor1, sensor2 = self.getErrorTableCode(mode)
                                    errorMode = (modelTime, sensor1, sensor2)
                                    self.imageCount += 1
                                    self.setTargetTimes(satId, targets)
                                    image = {"sat": satId, "time": tp, "cycle": cycleIndex, "image": self.imageCount,
                                             "targets": targets, "mode": {"type": mode, "errorMode": errorMode}}
                                    cycle["obs"].append(image)
                                    self.allImages.append(image)
                                    self.allRealImages.append(image)
                                else:
                                    removedImages.append({"sat": satId, "time": tp, "cycle": cycleIndex, "image": self.imageCount,
                                             "targets": targets, "mode": {"type": mode, "errorMode": errorMode}})
                        else: # downlink
                            # Assumption: No more than one DNL choice per second per sat (may have multiple GS targets)
                            # choice format: "12265: [{'cmd': 'DNL', 'targets': {'KangarooIsland', 'LockheedAus'}}]"
                            choice = None
                            for c in choices:
                                if c["cmd"] == "DNL":
                                    choice = c
                            assert choice, "createDataCycles() failed to find DNL choice in cmd: "+str(choices)
                            targets = self.getDnlTargets(choice)
                            for target in targets:
                                if target in self.gsList:
                                    if target in openDnlWindows:
                                        if tp == openDnlWindows[target]["end"] + 1:
                                            openDnlWindows[target]["end"] = tp
                                        else:  # discontinuous dnl window,  start new one for target
                                            openDnlWindows[target] = {"start": tp, "end": tp, "gs": target}
                                            cycle["dnl"].append(openDnlWindows[target])
                                    else:
                                        # create new dnl window for target
                                        openDnlWindows[target] = {"start": tp, "end": tp, "gs": target}
                                        cycle["dnl"].append(openDnlWindows[target])

            # create dummy end scene
            endScene = {"sat": satId, "time": self.planHorizon + 1 , "cycle": len(cycles)-1, "image": self.imageCount,  "endScene": True}
            cycles[-1]["obs"].append(endScene)
            self.allImages.append(endScene)
            satImageRange["lastImage"] = self.imageCount
            self.imageCount += 1

            self.allSatCycles[satId] = cycles
            print(f"createDataCycles() sat {satId} endScene: {endScene}")
            print(f"createDataCycles() sat {satId}, imageRange: {satImageRange}, end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.imageCount = len(self.allImages)
        print(f"removedImages: {len(removedImages)}")

    def isObsCmd(self, choice):
        cmd = choice["cmd"][0].lower()
        return cmd in ["p","l"]

    def isValidCmdAngle(self, cmd):
        sensor, angle = cmd.split(".")
        a = int(angle)
        if self.angleRange[0] <= a and a <= self.angleRange[1]:
            return True
        else:
            return False

    def isDnlCmd(self, choice):
        return choice["cmd"].lower().startswith("dnl")

    def choicesContainObsAndDnl(self, choices):
        obs = False
        dnl = False
        for choice in choices:
            type = choice["cmd"]
            if type == "obs":
                obs = True
            elif type == "DNL":
                dnl = True
        if obs and dnl:
            return True
        else:
            return False

    def setTargetTimes(self, satId, targets):
        for target in targets:
            if target not in self.targetTimes:
                self.targetTimes[target] = {}
            if satId not in self.targetTimes[target]:
                self.targetTimes[target][satId] = []
            self.targetTimes[target][satId].append(self.imageCount)

    def getObsTargets(self, cmd):
        targets = cmd["targets"]
        # for sourceId in cmd:
        #     targets.extend(cmd[sourceId])
        return sorted(targets)

    def getDnlTargets(self, cmd):
        return cmd["targets"]

    def addCycleStartAndEndTimes(self):
        for satId in self.allSatCycles:
            cycleStart = 0
            cycles = self.allSatCycles[satId]
            for cycle in cycles:
                cycle["start"] = self.getCycleStartTime(cycle)
                cycle["end"] = self.getCycleEndTime(cycle)

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
        filename = "latencies."+self.experimentDate+".txt"
        with open(filename, "w") as f:
            for sat in self.allSatCycles:
                f.write("Satellite "+sat+":\n")
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
        #TODO: Fix bug below which assumes only one target file name.
        #      Need to handle case with both pre and active fire targets
        filename = "selectedTargets."+self.experimentDate+"."+self.targetValFiles[0][3:-4]+".txt"
        totalSelectedTargetReward = 0
        for target in self.selectedTargets:
            val = self.targetValues[target]
            totalSelectedTargetReward += val
        print("Total selected target reward: "+str(round(totalSelectedTargetReward, 3)))
        with open(filename, "w") as f:
            f.write("# Selected target count: "+str(len(self.selectedTargets))+", total selected target reward: "+str(round(totalSelectedTargetReward, 3)))
            f.write("\n\n"+str(self.selectedTargets))

    def writeMissingActiveTargets(self):
        missingActiveTargets = []
        for activeTarget in self.activeFireTargets:
            if activeTarget not in self.selectedTargets:
                missingActiveTargets.append(activeTarget)
        #TODO: Fix bug below which assumes only one target file name.
        #      Need to handle case with both pre and active fire targets
        filename = "missingActiveTargets."+self.experimentDate+"."+self.targetValFiles[0][3:-4]+".txt"
        with open(filename, "w") as f:
            msg = "# Missing Active Fire Targets " + self.experimentDate + " (" + str(len(missingActiveTargets)) + ")"
            print(msg)
            f.write(msg)
            msg = "\n" + str(missingActiveTargets)
            print(msg)
            f.write("\n"+msg)

    def writeActiveFireTargets(self):
        filename = "activeFireTargets."+self.experimentDate+".txt"
        nzTargets = []
        for target in self.targetValues:
            if self.targetValues[target] > 0.0:
                nzTargets.append(target)
        with open(filename, "w") as f:
            f.write("# Active Fire Targets "+self.experimentDate+" ("+str(len(nzTargets))+")")
            f.write("\n\n"+str(nzTargets))

    def printRewardStats(self):
        allRewards = [self.targetValues[target] for target in self.targetTimes]
        sumRwd = sum(allRewards)
        maxRwd = max(allRewards)
        minRwd = min(allRewards)
        avgRwd = round(np.average(allRewards), 2)
        stdDevRwd = round(np.std(allRewards), 2)
        print("----\nAll Target Value Summary: Targets: " + str(len(allRewards)) + ", sum: " + str(
            sumRwd) + ", min: " + str(minRwd) + ", max: " + str(maxRwd) + ", avg: " + str(avgRwd) + ", stdDev: " + str(
            stdDevRwd) + "\n----\n")

    def printCycles(self):
        largeImageCount = 0
        smallImageCount = 0
        for satId in self.allSatCycles:
            satSelectedImageCount = 0
            cycles = self.allSatCycles[satId]
            print("\n-------------\nSat: " + str(satId) + ", cycles (" + str(len(cycles)) + "):")
            cycleId = 0
            imageQueue = []
            for cycle in cycles:
                obs = cycle["obs"]
                dnl = cycle["dnl"]
                selectedImages = cycle["selectedImages"] if "selectedImages" in cycle else None
                if selectedImages:
                    satSelectedImageCount += len(selectedImages)
                firstObsTime = obs[0]["time"] if obs else None
                lastObsTime = obs[-1]["time"] if obs else None
                firstImage, lastImage = self.firstAndLastScenesInCycle(cycle)
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
                # if lastObsTime and firstDnl:
                #     assert lastObsTime < firstDnl["start"], "printCycles() ERROR! obsTime "+str(lastObsTime)+" > dnlTime "+str(firstDnl)
                msg = "\n  Cycle "+str(cycleId)+" ["+str(cycle["start"])+" - "+str(cycle["end"])+"] ("+str(cycle["end"]-cycle["start"]+1)+")"
                msg += "    Obs: "+str(firstObsTime)+" - "+str(lastObsTime)+" ("+str(obsTicks)+"/"+str(obsDur)+" = "+str(obsPct)+" %), images: "+str(firstImage)+"-"+str(lastImage)
                # if selectedImages:
                #     for imageId in selectedImages:
                #         image = self.getImage(imageId)
                #         if image["type"] == "large":
                #             largeImageCount += 1
                #         else:
                #             smallImageCount += 1
                if firstDnl and lastDnl:
                    msg += "\n      DNL: "+str(firstDnl["start"])+" - "+str(lastDnl["end"]) + " ("+str(dnlTicks)+"/"+str(dnlDur)+" = "+str(dnlPct) +" %),  GS ("+str(len(cycle["dnl"]))+"):"
                    for slot in cycle["dnl"]:
                        dur = slot["end"] - slot["start"] + 1
                        msg += " ["+slot["gs"]+": "+str(slot["start"]) + " - "+str(slot["end"])+" ("+str(dur)+")]"
                print(msg)
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
                        sad = None
                        for s in dnlPlan["sad"]:
                            if s[0] == dnlSlot:
                                sad = s[1]
                        assignment = {"g": gs, "n": n, "t": tTime, "p": pTime, "s": sad}
                        dnlAssignments.append(assignment)
                    msg = "         DNL Plan: "
                    totalDnlDur = 0
                    sortedDnlAssignments = sorted(dnlAssignments, key=lambda x: x['t'])
                    for x in sortedDnlAssignments:
                        tTime = x["t"]
                        pTime = x["p"]
                        if tTime > 0.0 and pTime > 0.0:
                            endTime = tTime + pTime -1
                            dur = pTime
                            totalDnlDur += dur
                            msg += " [" + x["g"] + ": " + str(tTime) + " - " + str(endTime)+" ("+str(dur)+")]"
                    msg += " Total: "+str(totalDnlDur)
                    print(msg)

                if "availSpace" in cycle:
                    # TODO: BUG? should this be prior cycle data vs. current?
                    varIndex = "[" + satId + ", " + str(cycleId) + "]"
                    sa = round(cycle["availSpace"],2)
                    su = round(cycle["usedSpace"],2)
                    sf = round(cycle["freedSpace"],2)
                    rate = round(self.dnlRate,5)
                    downlinkTicks = round(sf / rate)
                    if self.includeEnergyConstraints:
                        ea = round(cycle["energyAvail"],2)
                        # eRaw = round(cycle["energyRaw"],2)
                        eNet = round(cycle["energyNet"],2)
                        powerIn = round(cycle["powerIn"],2)
                        powerOut = round(cycle["powerOutDefault"],2)
                        powerOutDnl = round(downlinkTicks * self.powerModel["powerOutDnlPct"],2)
                    selectedImageCount = len(selectedImages) if selectedImages else 0
                    dnlPlanSecs = cycle["dnlPlanSecs"] if "dnlPlanSecs" in cycle else 0
                    downlinkedImages = len(cycle["latencies"]) if "latencies" in cycle else 0
                    msg =     "      Plan: selected images: "+str(selectedImageCount)+", downlinked images: "+str(downlinkedImages)+", downlink secs: "+str(downlinkTicks) +", [dnlPlanSecs: "+str(dnlPlanSecs)+"]"
                    msg += "\n        sAvail "+str(sa)+" - sUsed "+str(su)+" + sFreed "+str(sf)+" = "+str(round(sa - su + sf,2))
                    msg += "\n          sad: "+str(dnlPlan["sad"])
                    msg += "\n          scd: "+str(dnlPlan["scd"])
                    msg += "\n          spd: "+str(dnlPlan["spd"])
                    if self.includeEnergyConstraints:
                        msg += "\n        eAvail " +str(ea)
                        msg += "\n        eNet "+str(eNet) +" = eAvail " +str(ea)+" +  eIn " +str(powerIn)+" - eOut "+str(powerOut) + " - eOutDnl "+ str(powerOutDnl)
                        # msg += "\n        eRaw "+str(eRaw) +" = eAvail " +str(ea)+" + eNet "+str(eNet)
                    print(msg)
                    images = []
                    if selectedImages:
                        print("Selected Images ("+str(len(selectedImages))+"):")
                        for id in selectedImages:
                            image = self.getImage(id)
                            images.append((id, image["time"]))
                        print(str(images))
                    if satId not in self.satCycleDetails:
                        self.satCycleDetails[satId] = []
                    details = {"satId": satId, "cycleId": cycleId, "cycle": cycle, "dnlAssignments": sortedDnlAssignments, "selectedImages": images}
                    self.satCycleDetails[satId].append(details)
                cycleId += 1
        totalRewards = 0
        missingTargets = []
        for target in self.targetTimes:
            if target in self.targetValues:
                val = self.targetValues[target]
                totalRewards += val
            else:
                missingTargets.append(target)
        print("\nTarget times: "+str(len(self.targetTimes))+", Target Values: "+str(len(self.targetValues))+", Total Available Target Rewards: "+str(round(totalRewards,3)))
        print("Large images: "+str(largeImageCount)+", Small images: "+str(smallImageCount))
        print(f"targets without rewards: {len(missingTargets)}")


    def writeCycleDetailsToFile(self, filename, cycleDetails, imageQueue):
       satId = cycleDetails["satId"]
       cycleId = cycleDetails["cycleId"]
       cycle = cycleDetails["cycle"]
       dnlAssignments = cycleDetails["dnlAssignments"]
       selectedImages = cycleDetails["selectedImages"]
       cycleStats = {"obsCount": 0, "dnlImageCount": 0, "dnlSecs": 0, "maxStoredImageCount": 0}
       cycleEvents = []
       for image in selectedImages:
           collectionTime = int(image[1])
           imageId = image[0]
           event = {"cmd": "obs", "eventTime": collectionTime, "image": imageId, "remainingPct": 100}
           cycleEvents.append(event)
           imageQueue.append(event)
       for dnlWindow in dnlAssignments:
           dnlWindowIndex = dnlWindow['n']
           dnlWindowStart = self.dnlSlotStart(satId, cycleId, dnlWindowIndex)
           dnlWindowEnd = self.dnlSlotEnd(satId, cycleId, dnlWindowIndex)
           dnlWindowDur = self.dnlSlotDuration(satId, cycleId, dnlWindowIndex)
           startTime = int(dnlWindow['t'])
           dur = int(dnlWindow['p'])
           cycleStats['dnlSecs'] += dur
           event = { "cmd": "dnl", "eventTime": startTime, "planDur": dur, "dnlWindowStart": dnlWindowStart, "dnlWindowEnd": dnlWindowEnd, "dnlWindowDur": dnlWindowDur, "gs": dnlWindow['g']}
           cycleEvents.append(event)
       sortedEvents = sorted(cycleEvents, key=lambda x: x['eventTime'])

       msg = ""
       if cycleId == 0:
           msg += satId
       msg += "\n_________________\n"
       currentBuffer = [i for i in imageQueue if i['eventTime'] < cycle["start"]]
       msg += f"Cycle {cycleId} [{cycle['start']} - {cycle['end']}] stored images: {len(currentBuffer)}\n\n"
       self.appendToFile(filename, msg)
       lineNumber = 0
       header = " Time  | Cmd       | # images |"+"\n"
       obsDuringDnl = []
       for event in sortedEvents:
           if lineNumber % 50 == 0:
               msg = "\n"+header
               self.appendToFile(filename, msg)
           if event["cmd"] == "obs":
               imageId = event["image"]
               if imageId not in obsDuringDnl:
                   collectionTime = int(event['eventTime'])
                   currentQueue = [i for i in imageQueue if i["eventTime"] <= collectionTime and "downlinkTime" not in i] #[i for i in imageQueue if i["eventTime"] <= collectionTime and "downlinkTime" not in i]
                   if currentQueue: # skip observations which occurred during dnlWindow (see printDnlDetails())
                       msg = f"{collectionTime}  | + OBS {event['image']} | {len(currentQueue)} |\n"
                       self.appendToFile(filename, msg)
                       cycleStats["obsCount"] += 1
                       if len(currentQueue) > cycleStats["maxStoredImageCount"]:
                           cycleStats["maxStoredImageCount"] = len(currentQueue)
           else:
               dnlWindowStart = event["dnlWindowStart"]
               dnlWindowEnd = event["dnlWindowEnd"]
               dnlWindowDur = event["dnlWindowDur"]
               planStart = int(event['eventTime'])
               planDur = int(event['planDur'])
               durPct = round(planDur/dnlWindowDur,2)
               msg = "\n" + str(dnlWindowStart) + " - " + str(dnlWindowEnd) + " (" + str(dnlWindowDur) + "), DNL  " + str(event['gs'])+ ", planStart: " + str(planStart) + ", planDur: " + str(planDur) + " (" + str(durPct) + "%)" + "\n"
               self.appendToFile(filename, msg)
               header = " Time  | Cmd        | # images |" + "\n"
               self.appendToFile(filename, header)
               imageQueue, obsDuringDnl, dnlImages, maxStoredImageCount = self.printDnlDetails(filename, satId, cycleId, cycle, planStart, planDur, dnlWindowEnd, imageQueue)
               cycleStats["obsCount"] += len(obsDuringDnl)
               cycleStats["dnlImageCount"] += len(dnlImages)
               if maxStoredImageCount > cycleStats["maxStoredImageCount"]:
                   cycleStats["maxStoredImageCount"] = maxStoredImageCount

           lineNumber += 1

       cycleStatsMsg = f"=== End of Cycle {cycleId}! images collected: {cycleStats['obsCount']}, images downlinked: {cycleStats['dnlImageCount']}, dnl secs: {cycleStats['dnlSecs']}, max stored images: {cycleStats['maxStoredImageCount']} ===\n"
       self.appendToFile(filename,cycleStatsMsg)
       if "latencies" in cycle:
           latencies = [image["latency"] for image in cycle["latencies"]]
           avg = str(round(float(np.average(latencies)), 2))
           std = str(round(float(np.std(latencies)), 2))
           mx = str(round(float(np.max(latencies)), 2))
           mn = str(round(float(np.min(latencies)), 2))
           latencyMsg = f"=== Cycle {cycleId} latencies ({len(latencies)}): avg: {avg}, min: {mn}, max: {mx}, stdDev: {std}\n"
           self.appendToFile(filename, latencyMsg)
       return imageQueue

    def printDnlDetails(self, filename, satId, cycleId, cycle, planStart, planDur, dnlWindowEnd, imageQueue):
        obsDuringDnl = []
        dnlImages = []
        maxImageCount = 0
        if imageQueue:
            buffer = [i for i in imageQueue if i["eventTime"] < planStart]
            if buffer:
                currentImage = buffer[0]
                # print("Current image: sat " + str(sat) + ", cycle " + str(cycleId) + ", " + str(
                #     currentImage)+", imageCount: "+str(len(buffer)))
                currentImageCollectionTime = currentImage["eventTime"]
                if currentImageCollectionTime < dnlWindowEnd:
                    processingTime = 0
                    processingComplete = False
                    for tick in range(planStart, dnlWindowEnd + 1):
                        scheduledObs = [i for i in imageQueue if i["eventTime"] == tick]
                        if scheduledObs:
                            assert len(scheduledObs) == 1, "printDnlDetails() ERROR multipleScheduledObs: "+str(scheduledObs)
                            obs = scheduledObs[0]
                            buffer.append(obs)
                            obsDuringDnl.append(obs["image"])
                            msg = f"{obs['eventTime']}  | + OBS {obs['image']} |  {len(buffer)}  |\n"
                            self.appendToFile(filename, msg)
                            if len(buffer) > maxImageCount:
                                maxImageCount = len(buffer)
                        elif currentImage and currentImageCollectionTime <= tick:
                            # subtract 5 % image remaining per second (20 seconds to downlink an image)
                            currentImage["remainingPct"] = round(currentImage["remainingPct"] - 5, 3)
                            processingTime += 1
                            if processingTime >= planDur:
                                processingComplete = True
                            if currentImage["remainingPct"] <= 0:
                                currentImage["downlinkTime"] = tick
                                latency = round(
                                    (currentImage["downlinkTime"] - currentImage["eventTime"] + 1) / 60, 2)
                                assert latency > 0, "calcualteSatLatencies() ERROR! Negative Latency: sat: " + str(
                                    satId) + ", cycle: " + str(cycleId) + ", image: " + str(
                                    currentImage) + ", latency: " + str(latency)
                                currentImage["latency"] = latency
                                if "latencies" not in cycle:
                                    cycle["latencies"] = []
                                downlinkedImage = buffer.pop(0)
                                dnlImages.append(downlinkedImage["image"])
                                imageQueue.pop(0)
                                cycle["latencies"].append(downlinkedImage)
                                imageTableInfo =  str(downlinkedImage['downlinkTime'])+ "  | - DNL "+ str(downlinkedImage["image"])+" |  "+str(len(buffer))+ "  |  latency "+ str(downlinkedImage["latency"])

                                # imageTableInfo = "| " + str(downlinkedImage["image"]).rjust(5, " ") + " | " + str(
                                #     downlinkedImage["start"]).rjust(9, " ") + " | " + str(
                                #     .rjust(10, " ") + " | " + str(
                                #     downlinkedImage["latency"]).rjust(7, " ") + " | " + str(len(buffer)).rjust(8,
                                #                                                                                " ") + "        |"
                                # print(imageTableInfo)
                                with open(filename, 'a') as f:
                                    f.write(imageTableInfo+"\n")
                                if buffer:
                                    currentImage = buffer[0]
                                    currentImageCollectionTime = currentImage["eventTime"]
                                else:
                                    currentImage = None
                                    msg = f"{tick}  | * Empty Buffer! * \n\n"
                                    # print(msg)
                                    self.appendToFile(filename, msg)
                                    continue
                            if processingComplete:
                                self.appendToFile(filename, f"{tick}  |  * end of plan dur *\n")
                                break
                    self.appendToFile(filename, f"{dnlWindowEnd}  | * end of dnl window *\n")
        return imageQueue, obsDuringDnl, dnlImages, maxImageCount

    def printCycleDetails(self):
        for satId in self.satCycleDetails:
            imageQueue = []
            filename = f"{satId}.cycles.txt"
            if os.path.exists(filename):
                os.remove(filename)
            for cycleDetails in self.satCycleDetails[satId]:
                imageQueue = self.writeCycleDetailsToFile(filename, cycleDetails, imageQueue)

    def appendToFile(self, filename, msg):
        with open(filename, 'a') as f:
            f.write(msg)

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

    def firstAndLastScenesInCycle(self, cycle):
        obs = cycle["obs"]
        first = obs[0]["image"] if obs else None
        last =  obs[-1]["image"] if obs else None
        return first, last

    # def lastSceneInSatCycle(self, sat, cycle):
    #     cycle = self.allSatCycles[sat][cycle]
    #     obs = cycle["obs"]
    #     return obs[-1]["image"] if obs else None

    def lastSceneInLastSatCycle(self, sat):
        lastCycle = self.allSatCycles[sat][-1]
        obs = lastCycle["obs"]
        return obs[-1]["image"] if obs else None

    def imagesWithModeContainingTarget(self, mode, target):
        matchingImages = []
        for image in self.allRealImages:
            if target in image["targets"]:
                imageErrorMode = image["mode"]["errorMode"]
                if mode == imageErrorMode:
                    matchingImages.append(image["image"])
        return matchingImages

    def imagesContainingTarget(self, target):
        targetOpportunities = self.targetTimes[target]
        images = []
        for sat in targetOpportunities:
            images.extend(targetOpportunities[sat])
        return images

#  Power model

    def combineEclipseFilesForSatSM(self, satId):
        print("combineEclipseFilesForSat() sat: "+str(satId))
        # combine sm eclipse files which are all 6 hours long and start at tick 0
        path = f"./inputs/SM/sat{satId}/s{satId}.eclipse_20200105T013000Z.csv"
        self.readEclipseFileForSatSM(satId, path,0)

    def readEclipseFileForSatSM(self, satId, file, initialTick):
        print(f"readEclipseFilesForSat() sat: {satId}, file: {file}, initialTick: {initialTick}")
        if satId not in self.eclipses:
            self.eclipses[satId] = set()
        satEclipses = self.eclipses[satId]
        assert os.path.exists(file), "readEclipseFileForSat() ERROR! file not found: "+file
        print("reading eclipse  file: "+file)
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("start"):
                    if line.count(",") > 0:
                        terms = line.split(",")
                        start = int(terms[0]) + initialTick
                        end   = int(terms[1]) + initialTick
                        # satEclipses.append((start, end))
                        eclipse = [x for x in range(start, end+1)]
                        satEclipses.update(eclipse)
                        if satId not in self.energyCheckpoints:
                            self.energyCheckpoints[satId] = [0]
                        self.energyCheckpoints[satId].append(end)

    def readEclipseFileForSat(self, satId):
        print("readEclipseFilesForSat() sat: "+str(satId))
        if satId not in self.eclipses:
            self.eclipses[satId] = set()
        satEclipses = self.eclipses[satId]
        path = "./inputs/"+self.experiment+"/operator/orbit_prediction/" + self.experimentRun + "/" + satId + "/eclipse/"
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
        path = self.dataPath+ "powerConfig.txt"
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

    def collectSummaryResults(self):
        # path = "./results/results.10.26.25/dnlOnlyWithObs.v3/dnlOnlyWithObs.cycle.8hr.v3"
        path = "./results/results.10.26.25/allGs.allDnl.v2/allGs.allDnl.cycle.1hr.v2"
        satStats = self.collectAggregateDnlStats(path)
        testResults = self.collectTestResults(path)
        targetInfo = testResults["targets"]
        availableTargets = str(targetInfo["totalCount"])
        selectedTargets = str(targetInfo["selectedCount"])
        targetPct = str(round(targetInfo["selectedCount"] / targetInfo["totalCount"], 2))
        availableTargetReward = str(targetInfo["availableTargetReward"])
        selectedTargetReward = str(targetInfo["selectedTargetReward"])
        targetRewardPct = str(round(targetInfo["selectedTargetReward"] / targetInfo["availableTargetReward"], 2))
        imageInfo = testResults["images"]
        availableImages = str(imageInfo["total"])
        selectedImages = str(imageInfo["selected"])
        imagePct = str(round(imageInfo["selected"] / imageInfo["total"], 2))
        pathTerms = path.split("/")
        filename = path+"/"+pathTerms[-1]+".resultSummary.txt"
        with (open(filename, "w") as f):

            f.write(f"Test results for: {path}\n")
            f.write(f"Cycle duration: {testResults['cycleDuration']} hours\n")
            f.write(f"Objective: {round(testResults['objective'],2)}, MIP gap: {testResults['gap']}\n")
            f.write(f"\nSummary (all sats):\n")
            f.write("------------------------------------------\n")
            f.write("Metric  |  Selected  | Available  |   %   |\n")
            f.write("------------------------------------------\n")
            msg =  f"Rewards | {selectedTargetReward.rjust(10)} | {availableTargetReward.rjust(10)} | {targetRewardPct.rjust(5)} |"
            f.write(msg+"\n")
            msg =  f"Targets | {selectedTargets.rjust(10)} | {availableTargets.rjust(10)} | {targetPct.rjust(5)} |"
            f.write(msg+"\n")
            msg =  f"Images  | {selectedImages.rjust(10)} | {availableImages.rjust(10)} | {imagePct.rjust(5)} |"
            f.write(msg+"\n")
            f.write("------------------------------------------\n\n")
            f.write("Latencies &  Max # of images in buffer:\n")
            f.write("--------------------------------------------------------------\n")
            f.write("Satellite |   Avg.   |   Max.  |    Min.  |   Std.   | images |\n")
            f.write("--------------------------------------------------------------\n")
            for satId in sorted(satStats.keys()):
                latencies = satStats[satId]['latencies']
                avg = str(latencies['avg'])
                max = str(latencies['max'])
                min = str(latencies['min'])
                std = str(latencies['std'])
                maxImagesInBuffer = str(satStats[satId]['maxStoredImages'])
                msg = f"{satId.rjust(9)} | {avg.rjust(8)} | {max.rjust(8)}| {min.rjust(8)} | {std.rjust(8)} |{maxImagesInBuffer.rjust(7)} |"
                f.write(msg+"\n")
            f.write("--------------------------------------------------------------\n")

    def collectTestResults(self,path):
        results = {"targets": {}}
        filename = path +"/plannerLog.txt"
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith("Target times:"):
                    print("haa")
                line = l.strip()
                if line.startswith("config: "):
                    line = line[8:]
                    config = ast.literal_eval(line)
                    results["cycleDuration"] = config["cycleDuration"]/3600
                    results["includeOnlyDnlOverlappingWithObs"] = config["includeOnlyDnlOverlappingWithObs"]
                elif line.startswith("Optimize a model with"):
                    terms = line.split(" ")
                    results["constraints"] = int(terms[4])
                    results["vars"] = int(terms[6])
                elif line.startswith("Best objective"):
                    terms = line.split(" ")
                    obj = float(terms[2][:-1])
                    results["objective"] = obj
                    results["gap"] = terms[7]
                elif line.startswith("Targets") and "unselected =" in line:
                    terms = line.split(" ")
                    targets = {"selectedCount": int(terms[1]), "unselectedCount": int(terms[4]), "totalCount": int(terms[7])}
                    results["targets"].update(targets)
                elif line.startswith("Images") and "unselected =" in line:
                    terms = line.split(" ")
                    targets = {"selected": int(terms[1]), "unselected": int(terms[4]), "total": int(terms[7])}
                    results["images"] = targets
                elif line.startswith("Total selected target reward"):
                    terms = line.split(" ")
                    results["targets"]["selectedTargetReward"] = float(terms[4])
                elif line.startswith("Total Available Rewards"):
                    terms = line.split(" ")
                    results["targets"]["availableTargetReward"] = float(terms[3])
        return results

    def collectAggregateDnlStats(self, path):
        cwd = os.getcwd()
        files = os.listdir(path)
        cycleFiles = [f for f in files if ".cycles.txt" in f]
        allSatInfo = {}
        for filename in cycleFiles:
            satId = filename.split(".")[0]
            latencies = []
            maxStoredImages = 0
            with open(filename, 'r') as f:
                lines = f.readlines()
                latencies = []
                for line in lines:
                    if line:
                        if "latency" in line:
                            latency = float(line.split(' ')[-1].strip())
                            latencies.append(latency)
                        if "max stored images" in line:
                            bufferSize = int(line.split(" ")[-2])
                            if bufferSize > maxStoredImages:
                                maxStoredImages = bufferSize

            avg =round(float(np.average(latencies)),2)
            std =round(float(np.std(latencies)),2)
            mx = round(float(np.max(latencies)),2)
            mn = round(float(np.min(latencies)),2)
            allSatInfo[satId] = {"latencies": {"avg": avg,"max": mx,"min":mn, "std":std}, "maxStoredImages": maxStoredImages}
        return allSatInfo

# *********** From old Soil Moisture planner ****

    def loadPreprocessingResults(self):
        self.readSlewTable()
        self.errorTable = self.readAllReducedErrorTables()
        self.initializeEvents()
        # self.convertHorizonVisibilityToChoiceFiles() # Creates Choices file from horizon visibility flat file
        self.initialHorizonGpErrAvg = self.getInitialHorizonGpErrAvg()
        if self.useSortedGP:
            self.sortAndCutHorizonGps() # returns only the top self.sortedGPpct # of targets

    def readSlewTable(self):
        filename = self.dataPath + "slewTable.txt"
        print("Reading Slew Table")
        self.maxCmdSetupTime = 0
        with open(filename, "r") as f:
            firstLine = True
            for line in f:
                if firstLine:
                    firstLine = False
                else:
                    terms = line.split(",")
                    poFrom = int(terms[0].strip())
                    poTo = int(terms[1].strip())
                    time = float(terms[2].strip())
                    energy = float(terms[3].strip())
                    if poFrom not in self.slewTable:
                        self.slewTable[poFrom] = {}
                    col = self.slewTable[poFrom]
                    col[poTo] = [time, energy]
                    if time > self.maxCmdSetupTime:
                        self.maxCmdSetupTime = time
        tableKeys = list(self.slewTable.keys())
        colCount = len(tableKeys)
        firstCol = self.slewTable[tableKeys[0]]
        rowCount = len(firstCol.keys())
        print("  slew table size: "+str(rowCount)+" x "+str(colCount))

    def readAllReducedErrorTables(self):
        errTable1  = self.readReducedErrorTable(1)
        errTable7  = self.readReducedErrorTable(7)
        errTable8  = self.readReducedErrorTable(8)
        errTable12 = self.readReducedErrorTable(12)
        errTable16 = self.readReducedErrorTable(16)
        return {1 : errTable1, 7: errTable7, 8: errTable8, 12: errTable12, 16: errTable16}

    def readReducedErrorTable(self, biomeType):
        errorTable = {}
        filepathIn = self.dataPath + "errorTable.igbp"+str(biomeType)+".txt"
        print("\nReading errorTable file: " + filepathIn)
        if not os.path.exists(filepathIn):
            print("\nreadReducedErrorTable() ERROR! File not found: " + filepathIn + "\n")
            return
        isFirstLine = True
        fileIn = open(filepathIn, "r")
        for line in fileIn:
            line = line.strip()
            if not line.startswith("#"):
                if isFirstLine:
                    isFirstLine = False
                else:
                    terms = line.split(",")
                    # print("terms: "+str(terms))
                    column1 = int(terms[0])
                    column2 = int(terms[1])
                    error = float(terms[2])
                    key = (column1, column2)
                    errorTable[key] = error
                    # print("errorTable[" + str(key) + "] = " + str(error))
        fileIn.close()
        return errorTable

    def initializeEvents(self):
        for satId in self.satList:
            self.readHorizonGpFile(satId)
            self.readHorizonVisibilityFile(satId)

    def readHorizonGpFile(self, satId):
        filename = self.getHorizonFilenamePrefix(satId, self.horizonId)+".gp.txt"
        filepath = self.dataPath + filename
        lineNumber = 0
        duplicateGP = 0
        print("readHorizonGpFile() file: "+str(filepath))
        assert os.path.exists(filepath), "readHorizonGpFile() ERROR! file not found "+filepath
        if os.path.exists(filepath):
            lineCount = 0
            print("reading gp file: "+filename)
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    pos = line.find(":")
                    line = line[pos+2:]
                    # print("ast line: "+str(line))
                    gpLine = ast.literal_eval(line)
                    gp = self.parseGpDict(gpLine)
                    newChoices = []
                    if gp.id in self.gpDict:
                        # print("readHorizonGpFile() ERROR! duplicate GP id: "+str(gp.id)+", new gp: "+str(gp)+", prior gp: "+str(self.gpDict[gp.id]))
                        duplicateGP += 1
                        oldErrorTable = self.gpDict[gp.id].errorTableChoices
                        newErrorTable = gp.errorTableChoices
                        for newChoice in newErrorTable:
                            isDuplicateChoice = False
                            for oldChoice in oldErrorTable:
                                if newChoice == oldChoice:
                                    # print("readHorizonGpFile() duplicate choice: "+str(newChoice))
                                    isDuplicateChoice = True
                            if not isDuplicateChoice:
                                newChoices.append(newChoice)
                        if newChoices:
                            # print("new choices: "+str(newChoices))
                            combinedErrorTable = []
                            combinedErrorTable.extend(oldErrorTable)
                            combinedErrorTable.extend(newChoices)
                            gp.errorTableChoices = combinedErrorTable
                    self.gpDict[gp.id] = gp
                    lineNumber += 1
                    if lineNumber % 10000 == 0:
                        print("  line: "+str(lineNumber))
            print("GP count: "+str(lineNumber) + ", duplicate GP count: "+str(duplicateGP))
        else:
            print("readHorizonGpFile() ERROR! file not found: "+filepath)

    def parseGpDict(self, dict):
        gp = GP(dict["gp"], None, None, "0")  # no lat, lon
        gp.type = dict["type"]
        if "rain" in dict:
            gp.rainHours = dict["rain"]
        if "accessTimes" in dict:
            gp.accessTimes = dict["accessTimes"]
        if "horizonAccessTimes" in dict:
            gp.horizonAccessTimes = dict["horizonAccessTimes"]
        if "filteredAccessTimes" in dict:
            gp.filteredAccessTimes = dict["filteredAccessTimes"]
        if "accessTimePairs" in dict:
            gp.accessTimePairs = dict["accessTimePairs"]
        if "pointingChoices" in dict:
            gp.pointingChoices = dict["pointingChoices"]
        if "errorChoices" in dict:
            gp.errorChoices = dict["errorChoices"]
        if "errorTableChoices" in dict:
            gp.errorTableChoices = dict["errorTableChoices"]
        if "initialModelError" in dict:
            gp.initialModelError = dict["initialModelError"]
        if "lat" in dict:
            gp.lat = dict["lat"]
        if "lon" in dict:
            gp.lon = dict["lon"]
        return gp

    def readHorizonVisibilityFile(self, satId):
        filename = self.getHorizonFilenamePrefix(satId, self.horizonId) + ".flat.txt"
        filepath = self.dataPath + filename
        print("\nreadHorizonVisibilityFile() Reading horizon file: " + filename)
        assert os.path.exists(filepath), "readHorizonVisibilityFile() ERROR! file not found "+filepath
        if not os.path.exists(filepath):
            print("\nreadHorizonVisibilityFile() ERROR! File not found: " + filepath + "\n")
            return
        lastDict = {}
        dictLines = ""
        lineCount = 0
        horizonEvents = {}
        totalTpChoices = 0
        maxTpChoices = 0
        maxTpChoicesTp = None
        with open(filepath, "r") as f:
            for line in f:
                lineCount += 1
                line = line.strip()
                # print("line: "+str(line))
                if not line.startswith("#"):
                    if line.startswith("{"):
                        dictLines = line
                    else:
                        dictLines += line
                    if line.endswith("}"):
                        d = ast.literal_eval(dictLines)
                        tp = list(d.keys())[0]
                        if self.maxTick and tp > self.maxTick:
                            break
                        choices = d[tp]
                        # NOTE:  tick = tp - 1 if self.imageLock else tp
                        choiceCombos = self.getChoiceCombos(choices)
                        choiceCount = len(choiceCombos)
                        totalTpChoices += choiceCount
                        if choiceCount > maxTpChoices:
                            maxTpChoices = choiceCount
                            maxTpChoicesTp = {"tp": tp, "choices": choices, "combos": choiceCombos}
                        horizonEvents[tp] = choices
                        dictLines = ""
                        # collect all unique gp
                        choiceKeys = list(choices.keys())
                        for choice in choiceKeys:
                            gpList = choices[choice]
                            for gpi in gpList:
                                self.horizonGPs.add(gpi)
                                if gpi not in self.gpDict:
                                    print("readFlatHorizonFile() ERROR! unknown gpi: "+str(gpi))
                                    gp = GP(gpi, None, None, "0") # no lat, lon, type
                                    self.gpDict[gpi] = gp

                        lastDict = {tp: choices}
                        if lineCount % 1000 == 0:
                            print("tp: " + str(tp))

        gpDictSize = len(self.gpDict.keys())
        tpCount = len(horizonEvents.keys())
        print("\ns"+str(satId)+" tp count: " + str(tpCount) + ", gp count: " + str(len(self.horizonGPs)) +", gpDict count: "+str(gpDictSize)+ ", lineCount: " + str(lineCount))
        print("\ns"+str(satId)+" tp choices: " + str(totalTpChoices) + ", avgChoices/tp: " + str(totalTpChoices/tpCount)+ ", max choices/tp: "+str(maxTpChoices))
        maxChoiceTp = maxTpChoicesTp["tp"]
        maxChoiceChoices = maxTpChoicesTp["choices"]
        maxChoiceCombos = maxTpChoicesTp["combos"]

        print("lastDict: " + str(lastDict))
        self.satEvents[satId] = horizonEvents

    def getInitialHorizonGpErrAvg(self):
        totalErr = 0
        errCount = 0
        for gpi in self.horizonGPs:
            modelErr = self.getGpModelErr(gpi, 0)
            totalErr += modelErr
            errCount += 1
        return totalErr/errCount

    def getGpModelErr(self, gp, tick):
        if isinstance(gp, int):
            gp = self.getGP(gp)
        modelTime = self.getModelTime(tick) # convert 6 hour horizon into 3 hour time indices used by soil model
        modelErrors = gp.initialModelError
        for time, err in modelErrors:
            if time == modelTime:
                return err

    def getModelTime(self, tick):
        ticksPerHour = 60*60
        if tick < 3 * ticksPerHour:
            return 0
        elif tick < 6 * ticksPerHour:
            return 3
        elif tick < 9 * ticksPerHour:
            return 6
        elif tick < 12 * ticksPerHour:
            return 9
        elif tick < 15 * ticksPerHour:
            return 12
        elif tick < 18* ticksPerHour:
            return 15
        elif tick < 21 * ticksPerHour:
            return 18
        else:
            return 21

    def getGP(self, index):
        if index in self.gpDict:
            return self.gpDict[index]

    def sortAndCutHorizonGps(self):
        gpPairs = []
        for gpi in self.horizonGPs:
            modelErr = self.getGpModelErr(self.getGP(gpi), 0)
            gpPairs.append((gpi, modelErr))
        gpPairs.sort(key=lambda x: x[1], reverse=True)
        count = len(gpPairs)
        first = gpPairs[0]
        last = gpPairs[count - 1]
        print(f"sortAndCutHorizonGps() original count: {count}, top {self.sortedGPpct} %, first: {first}, last: {last}" )
        # cut the last half out
        maxCount = int(count * self.sortedGPpct)
        gpPairs = gpPairs[:maxCount]
        count = len(gpPairs)
        first = gpPairs[0]
        last = gpPairs[count - 1]
        print("sortAndCutHorizonGps() final count: " + str(count) + ", first: " + str(first) + ", last: " + str(last))
        for gpPair in gpPairs:
            gpi = gpPair[0]
            err = gpPair[1]
            self.sortedHorizonGPs.append(gpi)
            self.sortedHorizonGPerr[gpi] = err

    def getHorizonFilenamePrefix(self, satId, hId, filter = None):
        hStart = ((hId - 1) * self.horizonDur) + 1 #21601 #1 #21601  #1
        hEnd = hStart + self.horizonDur - 1
        if not filter:
            filter = self.horizonFilter
        filename = f"sat{satId}/s{satId}.{hStart}-{hEnd}.{filter}"
        return filename

    def getChoiceCombos(self, choices):
        singleCmds = []
        doubleCmds = []
        for cmd in choices.keys():
            terms = cmd.split(".")
            singleCmds.append(terms)
        for cmd1 in singleCmds:
            for cmd2 in singleCmds:
                if cmd1 != cmd2:
                    i1 = cmd1[0]
                    i2 = cmd2[0]
                    a1 = cmd1[1]
                    a2 = cmd2[1]
                    if i1 == 'L' and i2 == 'P' and a1 == a2:
                        dblCmd = cmd1 + cmd2
                        doubleCmds.append(dblCmd)
        combos = singleCmds + doubleCmds
        return combos

    def convertHorizonVisibilityToChoiceFiles(self):
        for satId in self.satList:
            satEvents = self.satEvents[satId]
            filename = f"{self.dataPath}s{satId}_choices.txt"
            print("convertHorizonVisiblityToChoiceFiles() filename: "+filename)
            with open(filename, "w") as f:
                timepoints = list(satEvents.keys())
                for tp in timepoints:
                    tpCmds = satEvents[tp] # tpCmds = {'L.48': [1224808], 'L.49': [1223938, 1224807], 'P.48': [1223938, 1224807, 1224808], 'P.49': [1223938, 1224807, 1224808]}
                    choices = []
                    for cmd in tpCmds:
                        targets = tpCmds[cmd]
                        choices.append({'cmd': cmd, 'targets': targets})
                    choiceFileEntry = f"{tp}: {choices}\n"
                    f.write(choiceFileEntry)

    def getErrorTableCode(self, mode):
        # returns code for given pointingOption (code for 1 obs)
        sensor, angle = mode.split(".")
        angle = int(angle)
        code = 0
        if 28 <= angle and angle <= 35:
            code = 1
        elif (22 <= angle and angle <= 27) or (36 <= angle and angle <= 41):
            code = 2
        elif (14 <= angle and angle <= 21) or (42 <= angle and angle <= 49):
            code = 3
        if sensor == "L":
            return (code, 0)
        elif sensor == "P":
            return (0, code)

    def getErrorModeNew(self, mode):
        sensor, angle = mode.split(".")
        angle = int(angle)
        pointingOption  = 0
        if 30 <= angle < 40:
            pointingOption = 1
        elif 40 <= angle < 50:
            pointingOption = 2
        elif 50 <= angle < 60:
            pointingOption = 3
        errorMode = None


# Slewing
    def getSlewTimeAndEnergy(self, fromAngle, toAngle):
        slewTableRow = self.slewTable[fromAngle]
        slewTime, slewEnergy = slewTableRow[toAngle]
        slewTimeCeil = math.ceil(slewTime)
        return (slewTimeCeil, slewEnergy)


# DshieldPlanner().collectSummaryResults()
DshieldPlanner().run()


