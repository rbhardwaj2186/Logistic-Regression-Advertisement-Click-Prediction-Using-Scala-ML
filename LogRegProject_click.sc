import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{hour, col}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.log4j._

// Optional: Set the logging level to error
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark Session
val spark = SparkSession.builder()
  .appName("Logistic Regression Project")
  .master("local[*]")  // The * allows Spark to use all available cores on your machine
  .getOrCreate()

// Import implicits for the .as function
import spark.implicits._

// Use Spark to read in the Advertising CSV file
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("D:/Work/Gre/UTD/Courses/Fall_II/Scala/ScalaClassification_Project/ScalaClassificationProject/advertising.csv")

// Print the schema of the DataFrame
data.printSchema()

// Create a new column called Hour from the Timestamp containing the hour of the click
val timedata = data.withColumn("Hour", hour(col("Timestamp")))

// Select the relevant columns and rename "Clicked on Ad" to "label"
val logregdata = timedata.select(
  col("Clicked on Ad").as("label"),
  col("Daily Time Spent on Site"),
  col("Age"),
  col("Area Income"),
  col("Daily Internet Usage"),
  col("Hour"),
  col("Male")
)

// Create a new VectorAssembler object called assembler for the feature columns
val assembler = new VectorAssembler()
  .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour"))
  .setOutputCol("features")

// Use randomSplit to create a train/test split of 70/30
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Create a new LogisticRegression object called lr
val lr = new LogisticRegression()

// Create a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Fit the pipeline to the training set
val model = pipeline.fit(training)

// Get results on the test set
val results = model.transform(test)

// Convert the test results to an RDD using .as and .rdd
val predictionAndLabels = results.select(col("prediction"), col("label")).as[(Double, Double)].rdd

// Instantiate a new MulticlassMetrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Step 4: Get the confusion matrix
val confusionMatrix = metrics.confusionMatrix

// Step 5: Print the confusion matrix
println(s"Confusion matrix:\n$confusionMatrix")