
from sklearn.datasets import load_diabetes
from sklearn.linear_model import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

#Set random state
random_state=42

## Load the data
url_train = 'https://raw.githubusercontent.com/neal-logan/data/main/phishing-url-pirochet-train.csv'
phish = pd.read_csv(url_train)

#Create numeric target variable column
phish['y'] = phish['status'].replace('legitimate', 0).replace('phishing', 1)

#Drop unnecessary columns
phish = phish.drop(columns=['status','url'])

#X/y split
X = phish.drop(columns=['y'])
y = phish['y']

#Train/validation split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=random_state)

## Train a histogram gradient-boosting model
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

## Predict Y-pred values
y_pred = model.predict(X_validate)

## Print MSE
roc_auc = metrics.roc_auc_score(y_validate, y_pred)
print(roc_auc)

# Save the model
import joblib
joblib.dump(model, '/mnt/datalake/phishing_url_model.pkl')
