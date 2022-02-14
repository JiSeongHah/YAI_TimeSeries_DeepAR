import numpy as np

def calcArea(xLst,yLst):
    xArr = np.array(xLst)
    yArr = np.array(yLst)

    PlusAreaX, PlusAreaY = xArr[:-1],yArr[1:]
    PlusArea = np.dot(PlusAreaX,PlusAreaY)

    MinusAreaX, MinusAreaY = xArr[1:],yArr[:-1]
    MinusAre = np.dot(MinusAreaX,MinusAreaY)

    FinalArea = abs(PlusArea-MinusAre)/2

    return FinalArea


