#!/bin/bash
#../../start.sh
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part2/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/framingham.csv /part2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./framingham.py hdfs://$SPARK_MASTER:9000/part2/input/
#../../stop.sh




