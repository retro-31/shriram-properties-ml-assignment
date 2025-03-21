# House Price Prediction Assignment

This assignment involves building and deploying machine learning models to predict California house prices using the California Housing dataset.

## Objective

Develop, evaluate, and deploy multiple regression models to predict median house values (`MedHouseVal`).

## Dataset

**California Housing dataset** with features:

- `MedInc` (Median Income)
- `HouseAge` (Median house age)
- `AveRooms` (Average rooms per household)
- `AveBedrms` (Average bedrooms per household)
- `Population` (Total population in the block group)
- `AveOccup` (Average occupancy)
- `Latitude`
- `Longitude`

## Assignment Tasks

### 1. Data Preprocessing

- Loaded dataset using scikit-learn.
- Scaled numerical features with `StandardScaler`.

### 2. Model Development

Trained five regression models:

- Linear Regression
- Decision Tree
- Ridge Regression
- Neural Network (MLPRegressor)
- XGBoost

### 3. Model Evaluation

- models evaluated using RMSE, MAE, and R²-score.
- Stored evaluation metrics in `evaluation_results.json`.

### 4. Model Deployment

- Develop a Flask web application (`app.py`) that:
  - Displays evaluation metrics.
  - Accepts user input for the features.
  - Predicts median house prices in real-time.

## Project Structure

```bash
house_price_assignment/
├── models/
│   ├── LinearRegression.pkl
│   ├── DecisionTree.pkl
│   ├── Ridge.pkl
│   ├── NeuralNetwork.pkl
│   ├── XGBoost.pkl
│   └── scaler.pkl
├── evaluation_results.json
├── app.py
├── train.ipynb
└── templates/
    └── index.html
```

- `notebook.ipynb`: Data exploration, preprocessing, model training, evaluation.
- `models/`: Serialized models and scaler.
- `evaluation_results.json`: Performance metrics of the models.
- `app.py`: Flask application to interact with models.
- `templates/index.html`: Web interface for model interaction and predictions.

## Instructions to Run

1. **Clone repository and set up environment**:

```bash
conda create -n mlenv python=3.11
conda activate mlenv
conda install flask numpy pandas matplotlib seaborn scikit-learn jupyter notebook joblib json
```

2. **Run Jupyter Notebook** (optional exploration):

```bash
jupyter notebook
```

- Then run all the cells inside train.ipynb

3. **Launch Flask Web App**:

```bash
python app.py
```

Access the web app at: `http://127.0.0.1:5000/`