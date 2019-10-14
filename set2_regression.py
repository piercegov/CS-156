import numpy as np

numDatapoints = 10
runs = 1000
iterations = 10000

# PERCEPTRON STUFF
def updateWeights(point):
    weights[0] = weights[0] + (point[3] * point[0])
    weights[1] = weights[1] + (point[3] * point[1])
    weights[2] = weights[2] + (point[3] * point[2])
    return

def iteration(weights, datapoints):
    missidentified = []
    for datapoint in datapoints:
        if np.sign(sum(weights, datapoint)) != datapoint[3]:
            missidentified.append(datapoint)
    return missidentified

def getDisagreement(numPoints, weights, p1x, p1y, p2x, p2y):
    testData = []
    i = 0
    while (i < numPoints):
        x, y = uniformRandom(2)
        testData.append((1, x, y, makeDataPoint(x, y, p1x, p1y, p2x, p2y)))
        i += 1
    incorrect = iteration(weights, testData)
    return len(incorrect)/len(testData)
# END PERCEPTRON STUFF

def uniformRandom(numValues):
    return (np.random.uniform(-1, 1, numValues))

def sum(weights, values):
    res = (weights[0] * values[0]) + (weights[1] * values[1]) + (weights[2] * values[2])
    return res

def makeDataPoint(x, y, p1x, p1y, p2x, p2y):
    m = (p2y - p1y)/(p2x - p1x)
    if (y > m * (x - p1x) + p1y): return 1
    else: return -1

def setup():
    weights = [0, 0, 0]
    datapoints = []
    p1x, p1y, p2x, p2y = uniformRandom(4)

    for i in range(numDatapoints):
        x, y = uniformRandom(2)
        val = makeDataPoint(x, y, p1x, p1y, p2x, p2y)
        datapoints.append((1, x, y, val))
    return weights, datapoints, p1x, p1y, p2x, p2y

def seperatePoints(datapoints):
    points = []
    values = []
    for datapoint in datapoints:
        point = [datapoint[0], datapoint[1], datapoint[2]]
        points.append(point)
        values.append(datapoint[3])
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
        if pointSum > 0 and datapoint[3] < 0:
            missed.append(datapoint)
        elif pointSum < 0 and datapoint[3] > 0:
            missed.append(datapoint)
    return missed

totalInMissed = 0
totalOutMissed = 0
iterationArr = []
for r in range(runs):
    weights, datapoints, p1x, p1y, p2x, p2y = setup()
    points, values = seperatePoints(datapoints)
    weights = regression(points, values)
    missed = verifyPoints(weights, datapoints) # Testing in sample
    totalInMissed += len(missed)

    # Testing out of sample
    testpoints = []
    wrong = []
    for i in range(1000):
        # NEED THE ACTUAL FUNCTION
        x, y = uniformRandom(2)
        testpoints.append((1, x, y, makeDataPoint(x, y, p1x, p1y, p2x, p2y)))
        if (testpoints[i][3] != np.sign(sum(weights, testpoints[i][:3]))):
            wrong.append(testpoints[i])
    totalOutMissed += len(wrong)

    disagreements = [] # PLA
    for i in range(iterations):
        missidentified = iteration(weights, datapoints)
        if (len(missidentified) > 0):
            trainingPoint = missidentified[np.random.randint(0, len(missidentified))]
            updateWeights(trainingPoint)
        else:
            iterationArr.append(i)
            break

total = totalWrong = 0
for iteration in iterationArr:
    total += iteration

if len(iterationArr) != 0:
    print(len(iterationArr))
    print("Mean iteration #: " , total / len(iterationArr)) # Mean number of iterations required'