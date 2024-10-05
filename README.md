# Logistic Regression Advertisement Click Prediction Using Scala ML
 This project demonstrates a machine learning pipeline using Apache Spark's MLlib to predict whether a user will click on an advertisement based on user behavior and demographics. The model used in this project is Logistic Regression, a binary classification algorithm that predicts the likelihood of a user clicking an ad (Yes/No
Dataset

The dataset contains user data related to their daily internet usage, time spent on a website, demographic details such as age and income, and whether or not they clicked on an ad.
Key Features:

    Daily Time Spent on Site: Time spent by the user on the website.
    Age: Age of the user.
    Area Income: The average income in the user's area.
    Daily Internet Usage: Total time spent on the internet.
    Hour: Extracted from the timestamp when the user interacted with the website.
    Gender: Binary variable (Male or Female).

The target variable (label) is whether or not the user clicked on the advertisement (Clicked on Ad).
Project Workflow

    Data Loading: The dataset is loaded as a CSV file into a Spark DataFrame.
    Data Preprocessing: Features such as Hour are extracted from the timestamp, and relevant columns are selected for training.
    Data Splitting: The data is split into training (70%) and test (30%) sets.
    Model Training: A Logistic Regression model is trained using Apache Spark's MLlib.
    Model Evaluation: The performance of the model is evaluated on the test set, and a confusion matrix is generated to assess classification accuracy.

Technologies Used

    Apache Spark: For distributed data processing and machine learning.
    Scala: The primary programming language for this project.
    Apache Spark MLlib: For building the Logistic Regression model and feature processing.
    Spark SQL: For data manipulation.
    MulticlassMetrics: For evaluating model performance.

How to Run the Project
Prerequisites

    Apache Spark: Ensure that you have Apache Spark installed on your local machine. You can download it from here.
    Scala: Install Scala (version 2.12.x) to run the project.
    Java: Ensure that JDK 8 or higher is installed.
    IDE Setup: It's recommended to use an IDE like IntelliJ IDEA with the Scala plugin for development, or you can use the terminal for execution.

Steps to Run

    Clone the Repository:

    bash

git clone https://github.com/yourusername/Logistic-Regression-Advertisement-Click-Prediction.git
cd Logistic-Regression-Advertisement-Click-Prediction

Place the Dataset:

Download or place the dataset (advertising.csv) in the data/ directory of the project.

Run the Application:

    Using Spark-submit:

    bash

        spark-submit --class LogisticRegressionProject target/scala-2.12/logistic-regression-ad-click.jar

        Or, if using IntelliJ IDEA:
            Open the project.
            Create a new Scala object for the code.
            Run the application within the IDE.

Example Output

After running the project, the logistic regression model will produce a confusion matrix to evaluate how well the model performed on predicting advertisement clicks.
Confusion Matrix

The confusion matrix will help assess:

    True Positives (TP)
    True Negatives (TN)
    False Positives (FP)
    False Negatives (FN)

This matrix provides insight into the accuracy, precision, recall, and other performance metrics of the model.
Future Improvements

    Experiment with other classifiers like Decision Trees or Random Forests.
    Perform hyperparameter tuning to improve model performance.
    Incorporate additional features or engineer new features to enhance prediction accuracy.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Apache Spark MLlib documentation for guidance on using Spark for machine learning.
    The open-source community for providing resources and tools.
