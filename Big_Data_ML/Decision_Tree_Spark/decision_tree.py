from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array


def binary(x):
    if x == 'Y':
        return 1
    else:
        return 0

def mapEducation(degree):
    if degree == 'BS':
        return 1
    elif degree == 'MS':
        return 2
    elif degree == 'PhD':
        return 3
    else:
        return 0

def createLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = binary(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])

    return LabeledPoint(hired, array([yearsExperience, employed, previousEmployers, educationLevel, topTier, interned]))

# boilerplate spark stuff
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree") # run locally
sc = SparkContext(conf=conf)

# load data
rawData = sc.textFile("./PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)

# split lines into lists of features
csvData = rawData.map(lambda x: x.split(','))

# convert to labeld points
trainingData = csvData.map(createLabeledPoints)

# all decision trees in spark have to be labeld points and numeric data
# create test
testCandidates = [array([10, 1, 3, 1, 0, 0])]
testData = sc.parallelize(testCandidates) # make rdd

# category says number of categories at a certain index
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2}, impurity='gini', maxDepth=5, maxBins=32)

# nothing happens until action taken, which is here
predicitons = model.predict(testData)

print('Hire Prediction:')
results = predicitons.collect()
for result in results:
    print(result)

print('Learned classification tree model:')
print(model.toDebugString())
