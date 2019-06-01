#!/bin/bash
source ../../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /lab3/Q1/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /lab3/Q1/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /lab3/Q1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../../../data/shot_logs.csv /lab3/Q1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./MLlib.py hdfs://$SPARK_MASTER:9000/lab3/Q1/input/

