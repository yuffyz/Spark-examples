from __future__ import print_function
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, udf, sum
from pyspark.sql import types as T
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import *
import sys
import os
import math

os.environ["PYSPARK_PYTHON"] = "python2"
os.environ['PYTHONPATH'] = ':'.join(sys.path)

# converts sparse vector to dense vector
def sparse_to_array(v):
    v = DenseVector(v)
    new_array = list([float(x) for x in v])
    return new_array

# returns one-hot-encoded stages
def ohe(df):
    cols = df.columns
    # One-hot-encode category columns
    categoricalColumns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    # define income as label column
    label_stringIdx = StringIndexer(inputCol = 'income', outputCol = 'label')
    stages += [label_stringIdx]

    # vectorize numeric columns
    numericCols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="initial_features")
    stages += [assembler]

    # perform final pipeline to get vectorized feature column
    pipeline = Pipeline().setStages(stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'initial_features'] + cols
    df = df.select(selectedCols)

    # convert sparse vector initial_feature column to dense vector
    sparse_to_array_udf = udf(sparse_to_array, T.ArrayType(T.FloatType()))
    df = df.withColumn('dense_vector_features', sparse_to_array_udf('initial_features'))
    ud_f = udf(lambda r : Vectors.dense(r), VectorUDT())
    df = df.withColumn('features', ud_f('dense_vector_features'))
    return df

# returns a cleaned dataframe
def clean_df(df, freq_items):
    # drop unwanted columns
    columns_to_drop = ['education', 'fnlwgt']
    df = df.drop(*columns_to_drop)
    # replace missing values with most frequent values
    df = df.withColumn('native-country',
                       when(df['native-country'] == '?', freq_items[0][0][0]).otherwise(df['native-country']))
    df = df.withColumn('workclass',
                       when(df['workclass'] == '?', freq_items[0][1][0]).otherwise(df['workclass']))
    df = df.withColumn('occupation',
                       when(df['occupation'] == '?', freq_items[0][2][0]).otherwise(df['occupation']))
    # convert native-country to be a two value column
    df = df.withColumn('native-country',
                       when(df['native-country'] != 'United-States', 'Not-US').otherwise(df['native-country']))
    # one-hot-encode all categorical columns
    df = ohe(df)
    return df

# returns accuracy rate of given prediction dataframe
def get_accuracy_rate(pred_df):
    accuracy_rate = 0.0
    num_total_preds = pred_df.count()
    pred_df = pred_df.withColumn('isSame', when(pred_df['label'] == pred_df['prediction'], 1.0).otherwise(0.0))
    num_correct_preds = pred_df.select(sum('isSame')).collect()[0][0]
    accuracy_rate = (float(num_correct_preds) / float(num_total_preds)) * 100.0
    return accuracy_rate

reload(sys)
sys.setdefaultencoding('utf8')
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: p3.py <directory_of_files>", file=sys.stderr)
        sys.exit(-1)

    conf = SparkConf().setAppName("Project2Part3")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    directory_path = sys.argv[1]
    train_file_path = directory_path + '/adult.data.csv'
    test_file_path = directory_path + '/adult.test.csv'

    # read the train and test CSV datafiles into a dataframes
    train_df = sqlContext.read.load(train_file_path, format = 'com.databricks.spark.csv',
                                    header = 'true', inferSchema = 'true',
                                    ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')
    n_rows = train_df.count()
    n_cols = len(train_df.columns)
    train_df.show(5, False)
    print('# Initial Train Rows :', n_rows, '\t# Initial Train Columns :', n_cols, '\n')
    test_df = sqlContext.read.load(test_file_path, format = 'com.databricks.spark.csv',
                                   header = 'true', inferSchema = 'true',
                                   ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true')
    n_rows = test_df.count()
    n_cols = len(test_df.columns)
    test_df.show(5, False)
    print('# Initial Test Rows :', n_rows, '\t# Initial Test Columns :', n_cols, '\n')

    # cleaning train and test dataframes
    # finding frequent items in train_df to fill missing values in train and test dataframe
    freq_items = train_df.freqItems(['native-country', 'workclass', 'occupation'], support = 0.6).collect()
    train_df = clean_df(train_df, freq_items)
    train_df = train_df.withColumn('income', when(train_df['income'] == '<=50K', 0).otherwise(1))
    n_rows = train_df.count()
    n_cols = len(train_df.columns)
    train_df.show(5, False)
    train_df.printSchema()
    print('\n# Final Train Rows :', n_rows, '\t# Final Train Columns :', n_cols, '\n')

    test_df = clean_df(test_df, freq_items)
    test_df = test_df.withColumn('income', when(test_df['income'] == '<=50K.', 0).otherwise(1))
    n_rows = test_df.count()
    n_cols = len(test_df.columns)
    test_df.show(5, False)
    test_df.printSchema()
    print('\n# Final Test Rows :', n_rows, '\t# Final Test Columns :', n_cols, '\n')

    # create new dataframe with just two columns - features and income(label)
    vtrain_df = train_df.select(['label', 'features'])
    vtrain_df.show(5, False)
    vtrain_df.printSchema()
    print('\n\tFinal Changed Train Dataframe for Logistic Regression\n')
    vtest_df = test_df.select(['label', 'features'])
    vtest_df.show(5, False)
    vtest_df.printSchema()
    print('\n\tFinal Changed Test Dataframe for Logistic Regression\n')

    # initializing Logistic Regression Model
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter = 10)

    # fitting the model
    lrModel = lr.fit(vtrain_df)

    # printing the coefficients and intercept for logistic regression
    lr_coeff = lrModel.coefficients
    print('\nCoefficients: ')
    print([round(i, 3) for i in lr_coeff])
    print('\nIntercept: ', lrModel.intercept, '\n')

    # getting train predictions and accuracy rate / area under ROC / area under PR
    evaluator = BinaryClassificationEvaluator()
    train_pred = lrModel.transform(vtrain_df)
    train_pred.show(5, False)
    print('\nEntire Train Predictions DataFrame\n')
    train_roc = evaluator.setMetricName('areaUnderROC').evaluate(train_pred)
    train_pr = evaluator.setMetricName('areaUnderPR').evaluate(train_pred)
    condensed_train_pred = train_pred.select(['label', 'prediction'])
    train_acc = round(get_accuracy_rate(condensed_train_pred), 2)
    condensed_train_pred.show(10, False)
    print('\nCondensed Train Predictions DataFrame\n')
    print('Train Accuracy Rate :', train_acc)
    print('Train Area Under ROC :', round(train_roc, 4))
    print('Train Area Under PR :', round(train_pr, 4), '\n')

    # getting test predictions and accuracy rate
    test_pred = lrModel.transform(vtest_df)
    test_pred.show(5, False)
    print('\nEntire Test Predictions DataFrame\n')
    test_roc = evaluator.setMetricName('areaUnderROC').evaluate(test_pred)
    test_pr = evaluator.setMetricName('areaUnderPR').evaluate(test_pred)
    condensed_test_pred = test_pred.select(['label', 'prediction'])
    test_acc = round(get_accuracy_rate(condensed_test_pred), 2)
    condensed_test_pred.show(10, False)
    print('\nCondensed Test Predictions DataFrame\n')
    print('Test Accuracy Rate :', test_acc)
    print('Test Area Under ROC :', round(test_roc, 4))
    print('Test Area Under PR :', round(test_pr, 4), '\n')
