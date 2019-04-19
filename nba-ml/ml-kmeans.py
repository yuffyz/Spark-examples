from __future__ import print_function
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import sys

# need to have a file to process
if len(sys.argv) != 2:
    print("Usage: KMeans-ML-NBA <file>", file=sys.stderr)
    sys.exit(-1)


# player variables
players = ['james harden', 'chris paul','stephen curry','lebron james']


# -- Main Program -- 

# set up the session 
spark = SparkSession.builder.appName("KMeans-ML-NBA").getOrCreate()

# --------------------------------------------------------
# ---- load as pyspark dataframe for preprocessing -------

finalClusters = {p:-1 for p in players}

for player in players:
	# store file into a pyspark dataframe
	df = spark.read.format("csv").load(sys.argv[1], header="true", inferSchema="true")
	dataPts = df.filter(df.player_name == player).select('SHOT_DIST','CLOSE_DEF_DIST', 'SHOT_CLOCK').na.drop()

	# -----------------------------------------
	# ---------------- training ---------------

	# idx: [SHOT_CLOCK] = 8, [SHOT_DIST] = 11, [CLOSE_DEF_DIST] = 16
	assembler = VectorAssembler(inputCols=['SHOT_DIST','CLOSE_DEF_DIST', 'SHOT_CLOCK'], outputCol = "features")
	vector = assembler.transform(dataPts).select("features")

	# Trains a k-means model: 
	kmeans = KMeans().setK(4).setSeed(1)
	model = kmeans.fit(vector)
	finalClusters[player] = model.clusterCenters()

for player in players:
	print('Best Centroid for %s:' %(player), '\n', finalClusters[player])

spark.stop()


