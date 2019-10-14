import numpy as np

numDatapoints = 1000
runs = 1000

def closestHypothesis(datapoints, weights): # Tests 5 different possible hypothesis functions
    coeffA = (-1, -0.05, 0.08, 0.13, 1.5, 1.5)
    coeffB = (-1, -0.05, 0.08, 0.13, 1.5, 15)
    coeffC = (-1, -0.05, 0.08, 0.13, 15, 1.5)
    coeffD = (-1, -1.5, 0.08, 0.13, 0.05, 0.05)
    coeffE = (-1, -0.05, 0.08, 1.5, 0.15, 0.15)
    probA = probabilityOfHypothesis(datapoints, weights, coeffA)
    probB = probabilityOfHypothesis(datapoints, weights, coeffB)
    probC = probabilityOfHypothesis(datapoints, weights, coeffC)
    probD = probabilityOfHypothesis(datapoints, weights, coeffD)
    probE = probabilityOfHypothesis(datapoints, weights, coeffE)
    return (probA, probB, probC, probD, probE)


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
    weights = [-0.95694218, -0.05470223, 0.05077047, 0.00706217, 1.56600883, 1.46174011] # Hardcoded weights from the linear regression to test the hypothesis
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
totalA = totalB = totalC = totalD = totalE = 0 # Total for A, B, C, D, E
for r in range(runs):
    weights, datapoints = setup()
    points, values = seperatePoints(datapoints)
    # weights = regression(points, values)
    missed = verifyPoints(weights, datapoints) # Testing in sample (E_in)
    totalInMissed += len(missed)

    probA, probB, probC, probD, probE = closestHypothesis(datapoints, weights)

    totalA += probA # Could have used an array to make this prettier
    totalB += probB
    totalC += probC
    totalD += probD
    totalE += probE

    # Testing out of sample (E_out)
    testpoints = []
    wrong = []
    for i in range(1000):
        x, y = uniformRandom(2)
        newData = (transformData(x, y)) + (makeDataPoint(x, y),)
        testpoints.append(newData)
        if (testpoints[i][-1] != np.sign(sum(weights, testpoints[i][:-1]))):
            wrong.append(testpoints[i])
    totalOutMissed += len(wrong)

totalA = totalA / runs
totalB = totalB / runs
totalC = totalC / runs
totalD = totalD / runs
totalE = totalE / runs

print("A: " , totalA, " B: ", totalB, " C: " , totalC, " D: " , totalD, " E: " , totalE)
print(totalOutMissed / (runs * numDatapoints))