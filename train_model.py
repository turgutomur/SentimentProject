from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder \
    .appName("SentimentModelTrainer_Pro") \
    .master("local[*]") \
    .getOrCreate()

print("Veri seti yükleniyor...")


df = spark.read.csv("tweets_train.csv", inferSchema=True, header=False)
data = df.select("_c0", "_c5").toDF("raw_label", "text").dropna()


print("Veri seti %80 Train ve %20 Test olarak ayrılıyor...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print(f"Eğitim Verisi Sayısı: {train_data.count()}")
print(f"Test Verisi Sayısı:   {test_data.count()}")


indexer = StringIndexer(inputCol="raw_label", outputCol="label")
tokenizer = Tokenizer(inputCol="text", outputCol="words")


hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20000)
idf = IDF(inputCol="rawFeatures", outputCol="features")


lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])


print("Model eğitiliyor (Training)...")
model = pipeline.fit(train_data)


print("Test verisi üzerinde tahmin yapılıyor...")
predictions = model.transform(test_data)


evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("-" * 50)
print(f"🎯 MODEL BAŞARI RAPORU (TEST SETİ)")
print(f"✅ Doğruluk (Accuracy): %{accuracy * 100:.2f}")
print(f"❌ Hata Oranı (Test Error): %{(1.0 - accuracy) * 100:.2f}")
print("-" * 50)


model_path = "sentiment_model"
model.write().overwrite().save(model_path)
print(f"Model başarıyla '{model_path}' klasörüne güncellendi!")

spark.stop()
