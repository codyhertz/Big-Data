from pyspark.mllib.clustering import KMeans
from pyspark import SparkConf, SparkContext
from math import sqrt
import numpy as np
from sklearn.preprocessing import scale


k = 5

def createClusterdData(N, k):
    pointsPerCluster = float(N) / k

    x = []
    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)

        for j in range(int(pointsPerCluster)):
            x.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])

    x = np.array(x)
    return x

conf = SparkConf().setMaster("local").setAppName("SparkKMeans") # run locally
sc = SparkContext(conf=conf)

# scale normalizes the data
data = sc.parallelize(scale(createClusterdData(10000, k)))

clusters = KMeans.train(data, k, maxIterations=10, runs=10, initializationMode="random")

resultRDD = data.map(lambda point: clusters.predict(point)).cache()

print("Counts by value:")
counts = resultRDD.countByValue()
print(counts)

print("Cluster assignments: ")
results = resultRDD.collect()
print(results)

# evaluate clusters with within set sum of squared errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

wssse = data.map(lambda point: error(point)).reduce(lambda x, y: x + y) # takes error of each point and adds it all togethor
print("Within Set Sum of Squared Error = " + str(wssse))
