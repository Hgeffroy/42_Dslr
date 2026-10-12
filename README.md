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
### 0️⃣ Setup

```
pip install -r requirements.txt
```

All scripts default to the CSVs in `datasets/`, so they can be run without arguments. Use `-h` on any script to see its options.

### 1️⃣ Exploratory Analysis

Print statistical indicators (count, mean, std, min, quartiles, max) for each feature:
```
python3 src/describe.py -d datasets/dataset_train.csv
```

Generate histograms (saved in `figures/`):
```
python3 src/histogram.py -d datasets/dataset_train.csv -f "Care of Magical Creatures"
```

Generate scatter plots (saved in `figures/`):
```
python3 src/scatter_plot.py -d datasets/dataset_train.csv -f "Astronomy,Defense Against the Dark Arts"
```

Generate pair plots (saved in `figures/`):
```
python3 src/pair_plot.py -d datasets/dataset_train.csv
```

### 2️⃣ Training the Logistic Regression Model

Train the classifier:
```
python3 src/logreg_train.py -d datasets/dataset_train.csv
```
This will:
- Normalize the dataset
- Train one-vs-all logistic regression models
- Save learned parameters in `models/models.csv`

Optional flags: `-f` (comma-separated features), `-b` (batch size, `0` = full batch), `-i` (iterations), `-l` (learning rate).

### 3️⃣ Prediction

Use your trained model to predict houses for `dataset_test.csv` (written to `houses.csv`):
```
python3 src/logreg_predict.py -d datasets/dataset_test.csv -m models/models.csv
```

Plot the distribution of predictions (saved in `figures/`):
```
python3 src/draw_predictions.py -p houses.csv
```

## 📌 Concepts Covered

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

