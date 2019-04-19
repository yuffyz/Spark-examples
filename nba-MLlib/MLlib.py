from __future__ import print_function
from pyspark.mllib.clustering import KMeans, KMeansModel
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
spark = SparkSession.builder.appName("KMeans-MLlib-NBA").getOrCreate()


finalClusters = {p:-1 for p in players}

for player in players:

	# --------------------------------------------------------
	# ---- load the data for preprocessing -------
	df = spark.read.format("csv").load(sys.argv[1], header="true", inferSchema="true")
	dataPts = df.filter(df.player_name == player).select('SHOT_DIST','CLOSE_DEF_DIST', 'SHOT_CLOCK').na.drop()
	player_rdd = dataPts.rdd.map(lambda x: [float(x[0]), float(x[1]), float(x[2])])

	# -----------------------------------------
	# ---------------- training ---------------

	model = KMeans.train(player_rdd, 4, maxIterations=10)
	finalClusters[player] = model.clusterCenters

for player in players:
	print('Best Centroid for %s:' %(player), '\n', finalClusters[player])

spark.stop()

