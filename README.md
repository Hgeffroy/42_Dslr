# 🧬 42_DSLR  
*A mini data-science pipeline & logistic regression classifier built from scratch.*

## 🎯 Description  
The **DSLR** project (Data Science: Logistic Regression) from École 42 aims to introduce students to data analysis, visualization, and machine-learning fundamentals by implementing a full pipeline in Python—without external ML libraries.

You will:
- Explore, clean, and analyze a dataset  
- Visualize data using custom plots (histograms, scatter plots, pair plots)  
- Implement your own **logistic regression classifier** (binary or multi-class using one-vs-all)  
- Train your model through **gradient descent**  
- Evaluate performance using accuracy and confusion matrices  
- Apply the model to predict Hogwarts houses for unseen students  

The project focuses on understanding ML mechanics rather than relying on existing libraries.

## 🚀 Usage
### 1️⃣ Exploratory Analysis

Generate histograms:
```
python3 src/histogram.py data/dataset_train.csv
```

Generate scatter plots:
```
python3 src/scatter_plot.py data/dataset_train.csv
```

Generate pair plots:
```
python3 src/pair_plot.py data/dataset_train.csv
```

### 2️⃣ Training the Logistic Regression Model

Train the classifier:
```
python3 src/train.py data/dataset_train.csv
```
This will:
- Normalize the dataset
- Train one-vs-all logistic regression models
- Save learned parameters in weights/

### 3️⃣ Prediction

Use your trained model to predict houses for dataset_test.csv:
```
python3 src/predict.py data/dataset_test.csv
```

📌 Concepts Covered

- Data Cleaning & Normalization
- Exploratory Data Analysis (EDA)
- Statistical visualization:
  - Histograms
  - Scatter plots
  - Pair plots
- Logistic Regression:
  - Sigmoid function
  - Loss function (cross-entropy)
  - Gradient descent / Batch GD / Stochastic GD optimizations
  - One-vs-all classification
- Model evaluation:
  - Accuracy
  - Saving/loading trained models

