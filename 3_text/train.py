from __future__ import print_function
import sys

import pandas as pd
from pyspark.sql import SparkSession

# text processing packages
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.classification import LogisticRegression

# need to have a file to process
if len(sys.argv) != 2:
    print("Usage: Toxic Comment <file>", file=sys.stderr)
    sys.exit(-1)

# ----------- Main Program ----------

# set up the session 
spark = (SparkSession.builder.appName('Toxic Comment Classification').getOrCreate())

directory_path = sys.argv[1]
train_file_path = directory_path + '/train.csv'
test_file_path = directory_path + '/test.csv'

# ----------- Training ----------

# load the data as a Spark DataFrame
df = spark.read.format("csv").load(train_file_path, format = 'com.databricks.spark.csv',
                                    header = 'true', inferSchema = 'true',
                                    ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')

# change class values to int
df1 = df.withColumn('toxic', col('toxic').cast(IntegerType()))
df2 = df1.withColumn('severe_toxic', col('severe_toxic').cast(IntegerType()))
df3 = df2.withColumn('obscene', col('obscene').cast(IntegerType()))
df4 = df3.withColumn('threat', col('threat').cast(IntegerType()))
df5 = df4.withColumn('insult', col('insult').cast(IntegerType()))
df6 = df5.withColumn('identity_hate', col('identity_hate').cast(IntegerType()))

# clean the text
# get rid of new lines 
data = df6.select('id', (lower(regexp_replace('comment_text', "[^a-zA-Z\\s]", "")).alias('text')), 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
data = data.select('id', (regexp_replace('text', "[\r\n]+", "").alias('text')), 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
data.na.drop()
data.na.fill(0)
clean = data.where(col('toxic').isNotNull()).where(col('severe_toxic').isNotNull()).where(col('obscene').isNotNull()).where(col('threat').isNotNull()).where(col('insult').isNotNull()).where(col('identity_hate').isNotNull())

# Basic sentence tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_token = tokenizer.transform(clean)

# Remove stop words
remover = StopWordsRemover(inputCol='words', outputCol='words_clean')
df_words_no_stopw = remover.transform(words_token)

# term frequency
hashingTF = HashingTF(inputCol="words_clean", outputCol="rawFeatures")
tf = hashingTF.transform(df_words_no_stopw)

# tfidf
idf = IDF(inputCol = "rawFeatures", outputCol = "features")
idfModel = idf.fit(tf) 
tfidf = idfModel.transform(tf).select('features', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

# logistic regression 
REG = 0.1
lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)
lrModel = lr.fit(tfidf.limit(5000))
lr_train = lrModel.transform(tfidf)
lr_train.select("toxic", "probability", "prediction").show(20)

# --------------- Testing ---------------
# test set 

# load the data as a Spark DataFrame
test = spark.read.format("csv").load(test_file_path, format = 'com.databricks.spark.csv',
                                    header = 'true', inferSchema = 'true',
                                    ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')

# clean the text
# get rid of new lines 
test1 = test.select('id', (lower(regexp_replace('comment_text', "[^a-zA-Z\\s]", "")).alias('text')))
test2 = test1.select('id', (regexp_replace('text', "[\r\n]+", "").alias('text')))
test2.na.drop()
test2.na.fill(0)

test_tokens = tokenizer.transform(test2)
test_words_no_stopw = remover.transform(test_tokens)
test_tf = hashingTF.transform(test_words_no_stopw)
test_tfidf = idfModel.transform(test_tf)

# a udf to extract the probability of class 1: x[1]
extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
out_cols = [i for i in df.columns if i not in ["id", "comment_text"]]
test_res = test.select('id')

# fit the model to each label of the test data 
for col in out_cols:
    print(col)
    lr_test = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
    print("...fitting")
    model = lr_test.fit(tfidf)
    print("...predicting")
    res = model.transform(test_tfidf)
    print("...appending result")
    test_res = test_res.join(res.select('id', 'probability'), on = "id")
    print("...extracting probability")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
    test_res.show(20)

spark.stop()

