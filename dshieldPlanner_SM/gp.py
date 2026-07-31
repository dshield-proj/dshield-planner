# import dshieldUtil

class GP:
    def __init__(self, gpId, lat, lon, gpType):
        self.id = gpId
        self.lat = lat
        self.lon = lon
        self.type = int(gpType.strip()) # biome type
        self.viewed = False
        self.measurementError = 0.04
        self.initialModelError = [] # from predictor
        self.finalModelError = []   # from plan execution
        self.accessTimes = []
        self.horizonAccessTimes = []
        self.filteredAccessTimes = []
        self.rainAccessTimes = {}
        self.rainHours = []  # 0-based index of hour when rain threshold is met
        self.isSaturated = False
        self.accessTimePairs = None
        self.pointingChoices = None # choices based on pointing options
        self.errorChoices   = None # flattened choices based on error table codes ("Aspect 1")
        self.errorTableChoices = None
        self.planChoice = None
        self.yVars =[]

    def isRainTime(self, tp):
        hour = tp//3600 # convert seconds to hours with floor division
        if hour in self.rainHours:
            return True
        else:
            return False


    def prettyPrint(self):
        msg = "[gp "+str(self.id)
        type = self.biomeLabel(self.type) if isinstance(self.type, int) else self.type
        msg += " type: "+type #self.typeLabel()
        if self.rainHours:
            msg += ", rain: "+str(self.rainHours)
        if self.isSaturated:
            msg += ", saturated: "+str(self.viewed)
        if not self.measurementError == 0.04:
            msg += ", measurementErr: "+str(self.measurementError)
        if self.accessTimes:
            msg += ", accessTimes: "+str(sorted(self.accessTimes))
        if self.horizonAccessTimes:
            msg += ", horizonAccessTimes: "+str(self.horizonAccessTimes)
        if self.filteredAccessTimes:
            msg += ", filteredAccessTimes: "+str(self.filteredAccessTimes)
        if self.accessTimePairs:
            msg += ", accessTimePairs: "+str(self.accessTimePairs)
        if self.pointingChoices:
            msg += "\npointingChoices: "+str(self.pointingChoices)
        if self.errorChoices:
            msg += "\nerrorChoices: "+str(self.errorChoices)
        if self.errorTableChoices:
            msg += "\nerrorTableChoices ("+str(len(self.errorTableChoices))+"):\n"
            for choice in self.errorTableChoices:
                msg += str(choice)+"\n"
        if self.planChoice:
            msg += "planChoice: "+str(self.planChoice)
        else:
            msg += " * * UNPLANNED * *"
        msg += "\n\n"
        # if self.lat:
        #      msg += ", lat: "+str(self.lat)
        # if self.lon:
        #     msg += ", lon: " + str(self.lon)
        return msg

    def __str__(self):
        msg = "{'gp': "+str(self.id)

        msg += ", 'type': '"+self.biomeLabel(self.type)+"'"
        if self.lat:
             msg += ", 'lat': "+str(self.lat)
        if self.lon:
            msg += ", 'lon': " + str(self.lon)
        if self.rainHours:
            msg += ", 'rain': "+str(self.rainHours)
        if self.isSaturated:
            msg += ", 'saturated': "+str(self.viewed)
        if not self.measurementError == 0.04:
            msg += ", 'measurementErr': "+str(self.measurementError)
        if self.initialModelError:
            msg += ", 'initialModelError': "+str(self.initialModelError)
        if self.finalModelError:
            msg += ", 'finalModelError': "+str(self.finalModelError)
        if self.accessTimes:
            msg += ", 'accessTimes': "+str(sorted(self.accessTimes))
        if self.horizonAccessTimes:
            msg += ", 'horizonAccessTimes': "+str(self.horizonAccessTimes)
        if self.filteredAccessTimes:
            msg += ", 'filteredAccessTimes': "+str(self.filteredAccessTimes)
        if self.accessTimePairs:
            msg += ", 'accessTimePairs': "+str(self.accessTimePairs)
        if self.pointingChoices:
            msg += ", 'pointingChoices': "+str(self.pointingChoices)
        if self.errorChoices:
            msg += ", 'errorChoices': "+str(self.errorChoices)
        if self.errorTableChoices:
            msg += ", 'errorTableChoices': "+str(self.errorTableChoices)
        msg += "}"
        return msg

    def biomeTypeFromLabel(self, biomeLabel):
        # NOTE: DSHIELD ignores these GP types: 17=water, 11=wetlands, 13=urban, 15=frozen
        biomeLabel = biomeLabel.lower()
        if biomeLabel == "Evergreen Needleleaf Forest".lower():
            return 1
        elif biomeLabel == "Evergreen Broadleaf Forest".lower():
            return 2
        elif biomeLabel == "Deciduous Needleleaf Forest".lower():
            return 3
        elif biomeLabel == "Deciduous Broadleaf Forest".lower() or biomeLabel == "Deciduous Broadleaf Forrest".lower():
            return 4
        elif biomeLabel == "Mixed Forests".lower():
            return 5
        elif biomeLabel == "Closed Shrublands".lower():
            return 6
        elif biomeLabel == "Open Shrublands".lower():
            return 7
        elif biomeLabel == "Woody Savannas".lower():
            return 8
        elif biomeLabel == "Savannas".lower():
            return 9
        elif biomeLabel == "Grasslands".lower():
            return 10
        elif biomeLabel == "Wetlands".lower():  # ignored by DSHIELD
            return 11
        elif biomeLabel == "Croplands".lower():
            return 12
        elif biomeLabel == "Urban".lower():  # ignored by DSHIELD
            return 13
        elif biomeLabel == "Cropland and Natural Mosaic".lower():
            return 14
        elif biomeLabel == "Frozen".lower():  # ignored by DSHIELD
            return 15
        elif biomeLabel == "Bare".lower():
            return 16
        elif biomeLabel == "Water".lower():  # ignored by DSHIELD
            return 17
        else:
            print("biomeTypeFromLabel() ERROR! unknown label: " + biomeLabel)
            return 7

    def biomeLabel(self, type):
        # NOTE: DSHIELD ignores these GP types: 17=water, 11=wetlands, 13=urban, 15=frozen
        if type == 0:
            return "None"
        elif type == 1:
            return "Evergreen Needleleaf Forest"
        elif type == 2:
            return "Evergreen Broadleaf Forest"
        elif type == 3:
            return "Deciduous Needleleaf Forest"
        elif type == 4:
            return "Deciduous Broadleaf Forest"
        elif type == 5:
            return "Mixed Forests"
        elif type == 6:
            return "Closed Shrublands"
        elif type == 7:
            return "Open Shrublands"
        elif type == 8:
            return "Woody Savannas"
        elif type == 9:
            return "Savannas"
        elif type == 10:
            return "Grasslands"
        elif type == 11:  # ignored by DSHIELD
            return "Wetlands"
        elif type == 12:
            return "Croplands"
        elif type == 13:  # ignored by DSHIELD
            return "Urban"
        elif type == 14:
            return "Cropland and Natural Mosaic"
        elif type == 15:  # ignored by DSHIELD
            return "Frozen"
        elif type == 16:
            return "Bare"
        elif type == 17:  # ignored by DSHIELD
            return "Water"
        else:
            return type
