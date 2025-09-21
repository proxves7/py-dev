import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

### Classification Problem
cervical_data = pd.read_csv('data/cervical.csv')
print(cervical_data)
print(cervical_data.columns)

cervical_data.drop(['STDs..Time.since.first.diagnosis', 'STDs..Time.since.last.diagnosis'], inplace=True, axis=1)

cervical_data.info()

cervical_data['Biopsy'] = cervical_data['Biopsy'].map({'Healthy':0, 'Cancer': 1})
y = cervical_data.pop('Biopsy')
X = cervical_data.copy()

num_cols = ['Age', 'Number.of.sexual.partners', 'First.sexual.intercourse',
       'Num.of.pregnancies', 'Smokes..years.', 'Hormonal.Contraceptives..years.', 
       'IUD..years.', 'STDs..number.', 'STDs..Number.of.diagnosis']
cat_cols = ['Smokes', 'Hormonal.Contraceptives', 'IUD', 'STDs',]

X = pd.get_dummies(data=X, columns=cat_cols)
print(X)

class Cervical_DataLoader():
    def __init__(self):
        self.data = None
    
    def load_dataset(self, path="data/cervical.csv"):
        self.data = pd.read_csv(path)
    
    def preprocess_data(self):
        self.data.drop(['STDs..Time.since.first.diagnosis', 'STDs..Time.since.last.diagnosis'], inplace=True, axis=1)

        self.data['Biopsy'] = self.data['Biopsy'].map({'Healthy':0, 'Cancer': 1})

        num_cols = ['Age', 'Number.of.sexual.partners', 'First.sexual.intercourse',
            'Num.of.pregnancies', 'Smokes..years.', 'Hormonal.Contraceptives..years.', 
            'IUD..years.', 'STDs..number.', 'STDs..Number.of.diagnosis']
        cat_cols = ['Smokes', 'Hormonal.Contraceptives', 'IUD', 'STDs',]

        self.data = pd.get_dummies(data=self.data, columns=cat_cols)

    def get_data_split(self):
        y = self.data.pop('Biopsy')
        X = self.data.copy()
        return train_test_split(X, y, test_size=0.20, random_state=2022)
    
data_loader = Cervical_DataLoader()
data_loader.load_dataset()
data = data_loader.data

print(data.shape)
print(data.head())
data.info()

data_loader.preprocess_data()
print(data_loader.data.head())


##==================================================
### Regression Problem
bike_data = pd.read_csv('data/bike.csv')
print(bike_data)

bike_data.info()

y = bike_data.pop('cnt')
X = bike_data.copy()

num_cols = ['temp', 'hum', 'windspeed', 'days_since_2011']
cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

class Bike_DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/bike.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        num_cols = ['temp', 'hum', 'windspeed', 'days_since_2011']
        cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
        
        self.data = pd.get_dummies(data=self.data, columns=cat_cols)

    def get_data_split(self):
        y = self.data.pop('cnt')
        X = self.data.copy()
        return train_test_split(X, y, test_size=0.20, random_state=2022)

data_loader = Bike_DataLoader()
data_loader.load_dataset()
data = data_loader.data

print(data.shape)
print(data.head())
data.info()

data_loader.preprocess_data()
data_loader.data.head()
