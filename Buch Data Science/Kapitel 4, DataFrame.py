import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
#df = pd.read_csv(StringIO(csv_data))
#print(df)
#print(df.isnull().sum())
#print(df.dropna(subset=['C']))

#Ersetzen durch den Mittelwert der Spalte
#imr = SimpleImputer(missing_values=np.nan, strategy='mean')
#imr = imr.fit(df.values)
#imputed_data = imr.transform(df.values)
#print(imputed_data)

#### Beispiel Nominale zu Ordinale Merkmale ###
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 13.5, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
size_mapping = {'XL' : 3,
                'L' : 2,
                'M' : 1}
df['size'] = df['size'].map(size_mapping)
print(df)
# inv_size_mapping = {v: k for k, v in size_mapping.items()}
# df['size'] = df['size'].map(inv_size_mapping)
# print(df)
class_mapping = {label:idx for idx, label in enumerate(
    np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
ohe = OneHotEncoder(categories=0)
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
