from PIL import Image
import numpy as np
from scipy.special import expit

def initParams(layersNodes):

    params = {}

    for l in range(len(layersNodes)-1):

        params["w"+str(l+1)] = np.zeros((layersNodes[l+1],layersNodes[l]), dtype=np.float32)
        params["b"+str(l+1)] = np.zeros((layersNodes[l+1],1), dtype=np.float32)

    return params

def randomMiniBatches(X, Y, miniBatchSize):
    
    examples = X.shape[1]
    perm = list(np.random.permutation(examples))
    X = X[:,perm]
    Y = Y[:,perm]
    miniBatches = []

    for j in range(examples//miniBatchSize):

        miniBatchX = X[:,j*miniBatchSize:(j+1)*miniBatchSize]
        miniBatchY = Y[:,j*miniBatchSize:(j+1)*miniBatchSize]

        miniBatch = (miniBatchX,miniBatchY)
        miniBatches.append(miniBatch)

    if examples%miniBatchSize!=0:

        miniBatchX = X[:,-examples%miniBatchSize:]
        miniBatchY = Y[:,-examples%miniBatchSize:]

        miniBatch = (miniBatchX,miniBatchY)
        miniBatches.append(miniBatch)

    return miniBatches

def updateParamsWithGD(params, grads, lr):

    for l in range(len(params)//2):
        params["w"+str(l+1)] -= lr*grads["dw"+str(l+1)]
        params["b"+str(l+1)] -= lr*grads["db"+str(l+1)]

    return params

def initVelocity(params):

    v = {}
    
    for l in range(len(params)//2):

        v["dw"+str(l+1)] = np.zeros(params["w"+str(l+1)].shape, dtype=np.float32)
        v["db"+str(l+1)] = np.zeros(params["b"+str(l+1)].shape, dtype=np.float32)

    return v

def updateParamsWithMomentum(params, grads, v, beta, lr):

    for l in range(len(params)//2):

        v["dw"+str(l+1)] = beta*v["dw"+str(l+1)] + (1-beta)*grads["dw"+str(l+1)]
        params["w"+str(l+1)] -= lr*v["dw"+str(l+1)]

        v["db"+str(l+1)] = beta*v["db"+str(l+1)] + (1-beta)*grads["db"+str(l+1)]
        params["b"+str(l+1)] -= lr*v["db"+str(l+1)]

    return params, v

def initAdam(params):

    v = {}
    s = {}
    
    for l in range(len(params)//2):

        v["dw"+str(l+1)] = np.zeros(params["w"+str(l+1)].shape, dtype=np.float32)
        v["db"+str(l+1)] = np.zeros(params["b"+str(l+1)].shape, dtype=np.float32)
        s["dw"+str(l+1)] = np.zeros(params["w"+str(l+1)].shape, dtype=np.float32)
        s["db"+str(l+1)] = np.zeros(params["b"+str(l+1)].shape, dtype=np.float32)

    return v, s

def updateParamsWithAdam(params, grads, v, s, beta1, beta2, t, lr, epsilon):

    v_corrected = {}
    s_corrected = {}

    for l in range(len(params)//2):

        v["dw"+str(l+1)] = beta1*v["dw"+str(l+1)] + (1-beta1)*grads["dw"+str(l+1)]
        v_corrected["dw"+str(l+1)] = v["dw"+str(l+1)] / (1-beta1**t)

        s["dw"+str(l+1)] = beta2*s["dw"+str(l+1)] + (1-beta2)*np.square(grads["dw"+str(l+1)])
        s_corrected["dw"+str(l+1)] = s["dw"+str(l+1)] / (1-beta2**t)

        params["w"+str(l+1)] -= lr * (v_corrected["dw"+str(l+1)] / (np.sqrt(s_corrected["dw"+str(l+1)] + epsilon)))
        
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        v_corrected["db"+str(l+1)] = v["db"+str(l+1)] / (1-beta1**t)

        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.square(grads["db"+str(l+1)])
        s_corrected["db"+str(l+1)] = s["db"+str(l+1)] / (1-beta2**t)

        params["b"+str(l+1)] -= lr * (v_corrected["db"+str(l+1)] / (np.sqrt(s_corrected["db"+str(l+1)] + epsilon)))

    return params, v, s

def forwardProp(miniBatchX, params):

    caches = [{"w":[], "b":[], "z":None, "a":miniBatchX}]

    for l in range(len(params)//2):

        cache = {}
        cache["w"] = params["w"+str(l+1)]
        cache["b"] = params["b"+str(l+1)]
        zl = np.dot(cache["w"],caches[-1]["a"])+cache["b"]
        cache["z"] = zl
        al = expit(zl)
        cache["a"] = al
        caches.append(cache)

    return al, caches

def computeCost(al, miniBatchY, epsilon):

    miniBatchSize = miniBatchY.shape[1]
    cost = np.sum(-miniBatchY*np.log(al+epsilon)-(1-miniBatchY)*np.log(1-al+epsilon))

    return cost

def backwardProp(miniBatchX, miniBatchY, caches, epsilon):

    miniBatchSize = miniBatchX.shape[1]
    al = caches[-1]["a"]
    dal = (1/miniBatchSize)*np.divide(al-miniBatchY+epsilon, np.multiply(al+epsilon,1-al+epsilon))
    grads = {}

    for l in range(len(caches)-1,0,-1):

        al = caches[l]["a"]
        dzl = dal*(al*(1-al))
        grads["dw"+str(l)] = (1/miniBatchSize)*np.dot(dzl, caches[l-1]["a"].T)
        grads["db"+str(l)] = (1/miniBatchSize)*np.sum(dzl, axis=1, keepdims=True)

        dal = np.dot(caches[l]["w"].T, dzl)

    return grads

def model(X, Y, layersNodes, optimizer, lr, miniBatchSize, beta, beta1, beta2,  epsilon, numEpochs, printCost = True):

    numLayers = len(layersNodes)
    costs = []
    t = 1
    examples = X.shape[1]

    params = initParams(layersNodes)

    if optimizer=="gd":
        pass
    elif optimizer=="gd with momentum":
        v = initVelocity(params)
    elif optimizer=="adam":
        v, s = initAdam(params)

    for i in range(numEpochs):

        epochCost = 0
        miniBatches = randomMiniBatches(X, Y, miniBatchSize)

        for miniBatch in miniBatches:

            miniBatchX, miniBatchY = miniBatch
            al, caches = forwardProp(miniBatchX, params)
            miniBatchCost = computeCost(al, miniBatchY, epsilon)
            epochCost += miniBatchCost
            grads = backwardProp(miniBatchX, miniBatchY, caches, epsilon)

            if optimizer == "gd":
                params = updateParamsWithGD(params, grads, lr)
            elif optimizer == "gd with momentum":
                params, v = updateParamsWithMomentum(params, grads, v, beta, lr)
            elif optimizer == "adam":
                t += 1
                params, v, s = updateParamsWithAdam(params, grads, v, s, t, lr, beta1, beta2, epsilon)
        
        epochCost /= examples

        if printCost and i%1==0:
            print ("Cost after epoch %i: %f" % (i, epochCost))

    return params

X = ...
Y = ...

layersNodes = [X.shape[0], 48, 12, Y.shape[0]]
finalParams = model(X, Y, layersNodes, optimizer = "gd with momentum", lr = 0.1, miniBatchSize = 128, beta = 0.9, 
                    beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, numEpochs = 10, printCost = True)
