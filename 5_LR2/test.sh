#!/bin/bash
source ../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/input/
/usr/local/hadoop/bin/hdfs dfs -rm -r /part3/output/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/adult.data.csv /part3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/adult.test.csv /part3/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./p3.py hdfs://$SPARK_MASTER:9000/part3/input/
