import pandas as pd
import numpy as np
from datetime import datetime as dt


og = pd.read_csv('data/raw/SeoulBikeData.csv', encoding='cp1252')
df = og.copy()

df['Date'] = df['Date'].apply(lambda x: dt.strptime(x, '%d/%m/%Y'))
                              
df['Holiday'].replace(['No Holiday', 'Holiday'], [0, 1], inplace=True)
df['Functioning Day'].replace(['No', 'Yes'], [0, 1], inplace=True)

# Dew point temperature(°C) and Temperature(°C) have 0.91 of correlation, so we will drop one of them to avoid multicollinearity
df.drop('Dew point temperature(°C)', axis=1, inplace=True)

#df = pd.get_dummies(df)

df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day
df['WeekDay']=df['Date'].dt.day_name()
mapping_dictDay={'Monday': 1,'Tuesday': 2,'Wednesday': 3,'Thursday': 4,'Friday': 5,'Saturday': 6,'Sunday': 7}
df['WeekDayEncoding'] = df['WeekDay'].map(mapping_dictDay)
df['IsWeekend'] = np.where(df['WeekDayEncoding'] > 5, 1, 0)

df.to_csv('data/processed/SeoulBikeDataEDA.csv', date_format='%d/%m/%Y', index=False)
df.to_csv('data/processed/SeoulBikeDataEDA.csv', date_format='%d/%m/%Y', index=False)
