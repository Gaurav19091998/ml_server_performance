import pandas as pd
import numpy as np
df = pd.read_csv('DataCollector01.csv')
df = df.drop(0)
df = df.astype({"Time": str})
df['Time'] = df['Time'].str.replace(r'11/14/2022', '')
dataset = df.copy()
dataset = dataset.astype({"Memory": float,"Disk":float,"CPU":float})
dataset['performance'] = (dataset['Memory']+dataset['Disk']+dataset['CPU'])/3

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
dataset = train_set.drop("performance", axis=1)
dataset_label = train_set['performance'].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(fill_value=None)
dataset_w_t = dataset.drop("Time",axis = 1)
dataset_time = dataset['Time'].copy()
# SimpleImputer(missing_values=np.nan
imputer.fit(dataset_w_t)

X = imputer.transform(dataset_w_t)
dataset_tr = pd.DataFrame(X, columns=dataset_w_t.columns)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])

dataset_arr = my_pipeline.fit_transform(dataset_tr)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
model = DecisionTreeRegressor()
# model = LinearRegression()
model.fit(dataset_arr,dataset_label)

some_data = dataset_tr.iloc[:5]
some_labels = dataset_label.iloc[:5]

prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)

from sklearn.metrics import mean_squared_error
performance_predictions = model.predict(dataset_w_t)
lin_mse = mean_squared_error(dataset_label, performance_predictions)
lin_rms = np.sqrt(lin_mse)

x_test= test_set.drop({'Time','performance'}, axis = 1)
y_test = test_set['performance'].copy()
# x_test = 
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# unit_time_per = final_predictions/5
print(final_predictions[:1], y_test[:1]) 