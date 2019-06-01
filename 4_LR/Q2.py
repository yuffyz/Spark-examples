#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:10:54 2019

@author: TIANYING
"""
import sys
#import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,udf
from pyspark.sql import types as T
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
from pyspark.ml import Pipeline
#Import the vector assembler
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
#Scalarize the newly-created column 'feature'
from pyspark.ml.feature import StandardScaler
# Feature selection using chisquareSelector
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# converts sparse vector to dense vector
def sparse_to_array(v):
    v = DenseVector(v)
    new_array = list([float(x) for x in v])
    return new_array

# returns one-hot-encoded stages
def ohe(df):
    # One-hot-encode category columns
    categoricalColumns = ['education','currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    # vectorize numeric columns
    numericCols = ['male', 'age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]

    # perform final pipeline to get vectorized feature column
    pipeline = Pipeline().setStages(stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['TenYearCHD', 'features'] + cols
    df = df.select(selectedCols)

    # convert sparse vector initial_feature column to dense vector
    sparse_to_array_udf = udf(sparse_to_array, T.ArrayType(T.FloatType()))
    df = df.withColumn('dense_vector_features', sparse_to_array_udf('features'))
    ud_f = udf(lambda r : Vectors.dense(r), VectorUDT())
    df = df.withColumn('features', ud_f('dense_vector_features'))
    return df

def evaluation(TP, TN, FP, FN):
    print('\n\n', 'The confusion matrix is: \n\n', '[[',TN,FP,']\n\n','[',FN,TP,']]')
    #Model Evaluation - Statistics
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)

    print('\n\nThe acuuracy of the model = (TP+TN)/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',

    'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',

    'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n\n',

    'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n\n',

    'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n\n',

    'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n\n',

    'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n\n',

    'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity,'\n\n')


reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Q2.py <directory_of_files>", file=sys.stderr)
        sys.exit(-1)
        
    spark = SparkSession.builder.appName('lr-predic').getOrCreate()
    df = spark.read.csv(sys.argv[1], header = True, inferSchema = True)
    df.show()
    cols=df.columns
    df.printSchema()

    #Filter out all the 'NA' values
    df = df.filter((df.cigsPerDay != 'NA')& (df.BPMeds != 'NA') & (df.totChol != 'NA') & (df.BMI != 'NA') & (df.heartRate != 'NA') & (df.glucose != 'NA'))
    string_cols=['cigsPerDay', 'BPMeds','totChol','BMI', 'heartRate', 'glucose']
    for col_name in string_cols:
        df = df.withColumn(col_name, col(col_name).cast('float'))
    
    #Count of the class values
    df.groupby('TenYearCHD').count().show()

    #Drop the column of 'education'
    #Generative statistics conclusion
    cols.remove('education')
    cols.insert(0, 'Summary')
    df.describe().select(cols[:len(cols)//2]).show()
    df.describe().select(cols[len(cols)//2:-1]).show()

    #Use pipeline to combine all the features in one single feature vector
    cols.remove('TenYearCHD')
    cols.remove('Summary')
    print(cols)

    #['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 
    # 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

    df = ohe(df)
    df.printSchema()
    
    #Standarization
    standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    df=standardscaler.fit(df).transform(df)
    df.select("features","Scaled_features").show(5)
    
    #Split train & test
    train, test = df.randomSplit([0.8, 0.2], seed=12345)

    total=float(train.select("TenYearCHD").count())
    numPositives=train.select("TenYearCHD").where('TenYearCHD == 1').count()
    per_ones=(float(numPositives)/float(total))*100
    numNegatives=float(total-numPositives)
    print('\n\nThe number of Class 1 are {}'.format(numPositives))
    print('\n\nPercentage of Class 1 are {}'.format(per_ones))

    #Add balancing ratio
    BalancingRatio= numNegatives/total
    print('\n\nBalancingRatio = {}'.format(BalancingRatio))
    
    #Creat a new column named “classWeights” in the “train” dataset
    train=train.withColumn("classWeights", when(train.TenYearCHD==1, BalancingRatio).otherwise(1-BalancingRatio))
    train.select("classWeights", "TenYearCHD").show(10)
    
    #Feature selection
    css = ChiSqSelector(featuresCol='features',outputCol='Aspect',labelCol='TenYearCHD',fpr=0.05)
    train=css.fit(train).transform(train)
    test=css.fit(test).transform(test)
    test.select("Aspect").show(5,truncate=False)

    #Train the lr model
    lr = LogisticRegression(labelCol="TenYearCHD", featuresCol="Aspect",weightCol="classWeights", maxIter=10)
    model=lr.fit(train)

    #Printing the coefficients and intercept for logistic regression
    #lr_coeff=model.coefficients
    #print('\n\nCoefficients: ', [round(i, 3) for i in lr_coeff],'\n\n')

    predict_train=model.transform(train)
    predict_test=model.transform(test)
    predict_train.select("TenYearCHD","prediction").show(10)
    predict_test.select("TenYearCHD","prediction").show(10) 
    
    #Evaluation of the train model
    TP_train = predict_train.filter((predict_train.TenYearCHD == 1) & (predict_train.prediction == 1.0)).count()
    TN_train = predict_train.filter((predict_train.TenYearCHD == 0) & (predict_train.prediction == 0.0)).count()
    FP_train = predict_train.filter((predict_train.TenYearCHD == 0) & (predict_train.prediction == 1.0)).count()
    FN_train = predict_train.filter((predict_train.TenYearCHD == 1) & (predict_train.prediction == 0.0)).count()    
    print('\n\nEvaluation Results for Training Dataset')
    evaluation(TP_train, TN_train, FP_train, FN_train)
    
    #Evaluation of the test model
    TP_test = predict_test.filter((predict_test.TenYearCHD == 1) & (predict_test.prediction == 1.0)).count()
    TN_test = predict_test.filter((predict_test.TenYearCHD == 0) & (predict_test.prediction == 0.0)).count()
    FP_test = predict_test.filter((predict_test.TenYearCHD == 0) & (predict_test.prediction == 1.0)).count()
    FN_test = predict_test.filter((predict_test.TenYearCHD == 1) & (predict_test.prediction == 0.0)).count()
    print('\n\nEvaluation Results for Test Dataset')
    evaluation(TP_test, TN_test, FP_test, FN_test)

    
    #ROC for train & test
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="TenYearCHD")
    predict_test.select("TenYearCHD","rawPrediction","prediction","probability").show(5, truncate = False)

    print("\n\nThe area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
    print("\n\nThe area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))















