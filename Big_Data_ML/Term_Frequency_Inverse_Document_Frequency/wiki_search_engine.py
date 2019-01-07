from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import numpy as np


conf = SparkConf().setMaster("local").setAppName("SparkTFIDF") # run locally
sc = SparkContext(conf=conf)

rawData = sc.textFile("./subset-small.tsv")

fields = rawData.map(lambda x: x.split("\t")) # split into each document
documents = fields.map(lambda x: x[3].split(" ")) # split into words of the documents

documentNames = fields.map(lambda x: x[1])

# hash the words in the document
hashingTF = HashingTF(100000) # only 100000 words
tf = hashingTF.transform(documents) # turn words in documents into hashvalues

# compute tf*idf
tf.cache()
idf = IDF(minDocFreq=2).fit(tf) # ignore any word that doesnt appear at least twice
tfidf = idf.transform(tf)

# example using gettysburg
gettysburgTF = hashingTF.transform(['Gettysburg'])
gettysburgHashValue = gettysburgTF.indices[0]

# get tfidf score
gettysburgRelevence = tfidf.map(lambda x: x[np.asscalar(gettysburgHashValue)])

zippedResults = gettysburgRelevence.zip(documentNames)
print('Best document for Gettysburg is:')
print(zippedResults.max())
