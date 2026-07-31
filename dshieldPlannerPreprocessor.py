import os

class DshieldPlannerPreprocessor:
    def __init__(self, dataPath, orbitsPath, plannerOutputPath, satList, gsList, targetValues):
        self.satList = satList
        self.gsList = gsList
        self.dataPathRoot = dataPath
        self.orbitsPath = orbitsPath # from planner-config-20260626.json
        self.plannerOutputPath = plannerOutputPath  # from planner-config-20260626.json
        self.targetValues = targetValues
        self.satChoices = {}

    def start(self):
        for sat in self.satList:
            self.readSatTargetFile(sat)
            self.readSatGsFiles(sat)
            self.writeSatChoiceFile(sat)
        print("done")

    def readSatTargetFile(self, sat):
        satChoices = {}
        filepath = f"{self.dataPathRoot}{self.orbitsPath}{sat}/access/"
        assert os.path.exists(filepath), "readSatTargetFile() ERROR! path not found: "+filepath
        filenames = [x for x in os.listdir(filepath) if x.endswith(".csv")]
        if filenames:
            if len(filenames) > 1:
                print("readSatTargetFile() ERROR! multiple access files found in "+filepath+ ": "+str(filenames))
            else:
                filename = filenames[0]
        else:
            print("readSatTargetFile() ERROR! no access files found in "+filepath)
        header = [] # collect first 4 lines of file as the header
        print("readSatTargetFile() reading target file for "+sat+ ": "+filepath+filename)
        with open(filepath+filename, "r") as f:
            lineNumber = 0
            for line in f:
                line = line.strip()
                lineNumber += 1
                if 1 <= lineNumber and lineNumber <= 4:
                    header.append(line)
                    continue
                line = line.strip()
                if line:
                    tp, sourceId, gpList = line.split(" ")
                    tp = int(tp)
                    gpList = [int(gp) for gp in gpList.split(",")]
                    if tp not in satChoices:
                        satChoices[tp] = []  # [{"cmd": OBS/DNL, "targets": GP/GS}]
                    choices = satChoices[tp]
                    cmdChoice = None
                    for choice in choices:
                        if choice["cmd"] == "obs":
                            cmdChoice = choice
                            break
                    if not cmdChoice:
                        cmdChoice = {"cmd": "obs", "targets": []}
                        satChoices[tp].append(cmdChoice)
                    targets = [target for target in gpList if self.targetValues[target] > 0]
                    if targets:
                        cmdChoice["targets"].extend(targets)

        for tp in satChoices:
            choices = satChoices[tp]
            for cmd in choices:
                if cmd["cmd"] == "obs":
                    # remove duplicates & sort
                    targets = list(set(cmd["targets"]))
                    targets.sort()
                    cmd["targets"] = targets
        self.satChoices[sat] = satChoices

    def readSatGsFiles(self, sat):
        satChoices = self.satChoices[sat]# {TP: {"GS": [gsList]}}
        filepath = f"{self.dataPathRoot}{self.orbitsPath}{sat}/ground_contact/"
        # filepath = f"{self.dataPathRoot}orbits/sample/output/{sat}/ground_contact/"
        assert os.path.exists(filepath), "readSatGsFiles() ERROR! path not found: "+filepath
        filenames = os.listdir(filepath)
        assert filenames, "readSatGsFiles() ERROR! no files found found: "+filepath
        header = [] # collect first 4 lines of file as the header
        for filename in filenames:
            with open(filepath+filename, "r") as f:
                gsName = None
                lineNumber = 0
                for line in f:
                    line = line.strip()
                    lineNumber += 1
                    if 1 <= lineNumber and lineNumber <= 4:
                        if lineNumber == 1:
                            # strip off GS id from first line
                            gsName = line.split(" ")[-1]
                            if self.getGsId(gsName) not in self.gsList:
                                print("readSatGsFiles() skipping GS because its not in config gsList: "+str(gsName))
                                break
                            else:
                                print("readSatGsFiles() reading GS file for "+sat+ " GS "+ gsName+", file: "+filepath+filename)
                        header.append(line)
                        continue
                    line = line.strip()
                    if line:
                        start, end = line.split(",")
                        start = int(start)
                        end = int(end)
                        for tp in range(start, end+1):
                            # satChoices[tp] = {"DNL": None}
                            if tp not in satChoices:
                                satChoices[tp] = []  # [{"cmd": OBS/DNL, "targets": GP/GS}]
                            choices = satChoices[tp]
                            cmdChoice = None
                            for choice in choices:
                                if choice["cmd"] == "DNL":
                                    cmdChoice = choice
                                    break
                            if not cmdChoice:
                                cmdChoice = {"cmd": "DNL", "targets": set()}
                                satChoices[tp].append(cmdChoice)
                            cmdChoice["targets"].add(self.getGsId(gsName))
        self.satChoices[sat].update(satChoices)

    def getGsId(self, gsName):
        id = {"KangarooIsland": "KI",
              "LockheedAus": "AUS-L",
              "MerrittIsland": "MI",
              "PanoramaHeights": "PH"}
        return id[gsName] if gsName in id else gsName

    def writeSatChoiceFile(self, sat):
        filepath = self.dataPathRoot + self.plannerOutputPath
        if not os.path.exists(filepath):
            print("writeSatChoiceFile() creating dir: "+filepath)
            os.mkdir(filepath)
        filename = f"{filepath}{sat}_choices.txt"
        tpChoices = self.satChoices[sat]

        sortedTpChoices = sorted(tpChoices.keys())
        priorTP = None
        print(f"writeSatChoiceFile() {filename}\n")
        with open(filename, "w") as f:
            for tp in sortedTpChoices:
                if priorTP and tp - priorTP > 1:
                    diffSecs = tp - priorTP
                    gapSize = str(diffSecs)+"s" if diffSecs < 60 else str(round(diffSecs/60, 2))+"m"
                    f.write("\n--- GAP "+str(gapSize)+" ---\n")
                priorTP = tp
                tpCmdChoices = tpChoices[tp]
                filteredChoices = []
                for cmdChoice in tpCmdChoices:
                    if cmdChoice['targets']: # filter out any choices where all targets have value 0
                        filteredChoices.append(cmdChoice)
                if filteredChoices:
                    f.write(str(tp)+": "+str(filteredChoices)+"\n")


