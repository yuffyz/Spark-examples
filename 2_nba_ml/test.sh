#!/bin/bash
source ../../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /user/root/lab3/Q2/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /user/root/lab3/Q2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /user/root/lab3/Q2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../../../data/shot_logs.csv /user/root/lab3/Q2/input/shot_logs.csv
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./ml-kmeans.py hdfs://$SPARK_MASTER:9000//lab3/Q2/input/
