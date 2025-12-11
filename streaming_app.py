from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel


spark = SparkSession.builder \
    .appName("TwitterSentimentAnalysis") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1") \
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1:27017/TwitterDB.predictions") \
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1:27017/TwitterDB.predictions") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


print("Model yükleniyor...")
model = PipelineModel.load("sentiment_model")


df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "twitter-sentiment") \
    .load()


schema = StructType([
    StructField("text", StringType()), 
    StructField("label", IntegerType())
])
parsed_df = df.selectExpr("CAST(value AS STRING)").select(from_json(col("value"), schema).alias("data")).select("data.*")


prediction_df = model.transform(parsed_df)


final_df = prediction_df.withColumn("Duygu", when(col("prediction") == 0.0, "NEGATİF").otherwise("POZİTİF")) \
                        .select("text", "label", "Duygu", "prediction")


def write_to_mongo_and_console(batch_df, batch_id):
    
    batch_df.show(truncate=False)
    
    
    batch_df.write \
        .format("mongodb") \
        .mode("append") \
        .save()


print("Streaming ve MongoDB kaydı başladı...")
query = final_df.writeStream \
    .foreachBatch(write_to_mongo_and_console) \
    .start()

query.awaitTermination()
