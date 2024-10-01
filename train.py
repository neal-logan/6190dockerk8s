
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
infrared_thermography_temperature = fetch_ucirepo(id=925) 
  
# data (as pandas dataframes) 
X = infrared_thermography_temperature.data.features 
y = infrared_thermography_temperature.data.targets 


#Use sklearn preprocessing pipeline with one-hot encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

#Identify categorical columns
categorical_columns = ['Gender', 'Age', 'Ethnicity']

#Build the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(), categorical_columns)
])

#Establish model parameters
model_parameters = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42
}

#Build the preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(**model_parameters))
])

#Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit the pipeline on the training data
pipeline.fit(X_train, y_train['aveOralM'])


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#Predict and evaluate error on the test data
# y_pred = pipeline.predict(X_test)

# mse = mean_squared_error(y_test['aveOralM'], y_pred)
# print(mse)

# mae = mean_absolute_error(y_test['aveOralM'], y_pred)
# print(mae)

# Save the model
import joblib
joblib.dump(pipeline, '/mnt/datalake/ir_thermography_model.pkl')
