import numpy as np

def run_iteration(num_coins, num_flips): # Performs one 'run' of the experiment
    arr = np.array(np.zeros(num_coins))
    for i in range(num_coins):
        headcount = 0
        for j in range(num_flips):
            headcount += np.random.choice([0, 1])
        arr[i] = headcount
    return arr

def getCoins(arr): # Gets the 3 coins we care about from the array
    c1 = arr[0]
    crand = arr[np.random.randint(0, len(arr))]
    cmin = 100000
    for i in arr:
        if i < cmin:
            cmin = i
    return (c1, crand, cmin)

v1 = vrand = vmin = 0
num_runs = 100

for i in range(num_runs):
    arr = run_iteration(1000, 10)
    c1, crand, cmin = getCoins(arr)

    v1 += c1/10
    vrand += crand/10
    vmin += cmin/10

v1 = v1 / num_runs
vrand = vrand / num_runs
vmin = vmin / num_runs

print('V1: ' , v1, ', VRAND: ' , vrand, ' VMIN: ' , vmin)
