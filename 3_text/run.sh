#!/bin/bash
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /project2/P1/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /project2/P1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/train.csv /project2/P1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/test.csv /project2/P1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./p1.py hdfs://$SPARK_MASTER:9000/project2/P1/input/