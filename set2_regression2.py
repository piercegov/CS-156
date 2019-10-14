import numpy as np

numDatapoints = 1000
runs = 1000

def closestHypothesis(datapoints, weights):
    coeffA = (-1, -0.05, 0.08, 0.13, 1.5, 1.5)
    coeffB = (-1, -0.05, 0.08, 0.13, 1.5, 15)
    coeffC = (-1, -0.05, 0.08, 0.13, 15, 1.5)
    coeffD = (-1, -1.5, 0.08, 0.13, 0.05, 0.05)
    coeffE = (-1, -0.05, 0.08, 1.5, 0.15, 0.15)
    pA = probabilityOfHypothesis(datapoints, weights, coeffA)
    pB = probabilityOfHypothesis(datapoints, weights, coeffB)
    pC = probabilityOfHypothesis(datapoints, weights, coeffC)
    pD = probabilityOfHypothesis(datapoints, weights, coeffD)
    pE = probabilityOfHypothesis(datapoints, weights, coeffE)
    return (pA, pB, pC, pD, pE)


def probabilityOfHypothesis(datapoints, weights, coefficients):
    numWrong = 0
    for point in datapoints:
        if np.sign(sum(weights, point[:-1])) != np.sign(sum(coefficients, point[:-1])):
            numWrong += 1
    return 1 - (numWrong / len(datapoints))

def uniformRandom(numValues):
    return (np.random.uniform(-1, 1, numValues))

def sum(weights, values):
    res = 0
    for i in range(len(weights)):
        res += weights[i] * values[i]
    return res

def makeDataPoint(x, y):
    if (np.random.random() - 0.9 > 0):
        return -np.sign(x**2 + y**2 - 0.6)
    return np.sign(x**2 + y**2 - 0.6)

def transformData(x, y):
    return (1, x, y, x*y, x**2, y**2)

def setup():
    weights = [-0.95694218, -0.05470223, 0.05077047, 0.00706217, 1.56600883, 1.46174011]
    datapoints = []
    for i in range(numDatapoints):
        x, y = uniformRandom(2)
        val = makeDataPoint(x, y)
        newPoint = transformData(x, y) + (val,)
        datapoints.append(newPoint)
    return weights, datapoints

def seperatePoints(datapoints):
    points = []
    values = []
    for datapoint in datapoints:
        point = datapoint[:-1]
        points.append(point)
        values.append(datapoint[-1])
    return points, values

def regression(points, values):
    xMatrix = np.matrix(points)
    yMatrix = np.matrix(values).getT()
    pinv = np.linalg.pinv(xMatrix)
    return np.matmul(pinv, yMatrix)

def verifyPoints(weights, datapoints):
    missed = []
    for datapoint in datapoints:
        pointSum = sum(weights, datapoint)
        if pointSum > 0 and datapoint[-1] < 0:
            missed.append(datapoint)
        elif pointSum < 0 and datapoint[-1] > 0:
            missed.append(datapoint)
    return missed

totalInMissed = 0
totalOutMissed = 0
tA = tB = tC = tD = tE = 0 # Total for A, B, C, D, E
for r in range(runs):
    weights, datapoints = setup()
    points, values = seperatePoints(datapoints)
    # weights = regression(points, values)
    missed = verifyPoints(weights, datapoints) # Testing in sample
    totalInMissed += len(missed)

    pA, pB, pC, pD, pE = closestHypothesis(datapoints, weights)

    tA += pA
    tB += pB
    tC += pC
    tD += pD
    tE += pE

    # Testing out of sample
    testpoints = []
    wrong = []
    for i in range(1000):
        x, y = uniformRandom(2)
        newData = (transformData(x, y)) + (makeDataPoint(x, y),)
        testpoints.append(newData)
        if (testpoints[i][-1] != np.sign(sum(weights, testpoints[i][:-1]))):
            wrong.append(testpoints[i])
    totalOutMissed += len(wrong)

tA = tA / runs
tB = tB / runs
tC = tC / runs
tD = tD / runs
tE = tE / runs

print("A: " , pA, " B: ", pB, " C: " , pC, " D: " , pD, " E: " , pE)
print(totalOutMissed / (runs * numDatapoints))