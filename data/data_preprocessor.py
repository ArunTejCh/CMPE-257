import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


df = pd.read_csv("DJI_5_years.csv")
df['Next_Day'] = pd.Series(df['Close'], index=df.index)
df['Close-1'] = pd.Series(df['Close'], index=df.index)
df['Close-2'] = pd.Series(df['Close'], index=df.index)
df['Close-3'] = pd.Series(df['Close'], index=df.index)
df['Close-4'] = pd.Series(df['Close'], index=df.index)
print(df.head())
for i in range(4, len(df) - 1):
    df.at[i, 'Next_Day'] = df.iloc[i+1]['Close']
    df.at[i, 'Close-1'] = df.iloc[i - 1]['Close']
    df.at[i, 'Close-2'] = df.iloc[i - 2]['Close']
    df.at[i, 'Close-3'] = df.iloc[i - 3]['Close']
    df.at[i, 'Close-4'] = df.iloc[i - 4]['Close']

df = df[["Close-4", "Close-3", "Close-2", "Close-1", "Close", "Next_Day"]].copy()
df = df.iloc[4:]
print(df.head())
#df = df.reshape(-1, 1)

# Print shape of dataset
print(df.shape)


# Plot raw data
plt.figure(figsize=(12,5), frameon=False, facecolor='brown')
plt.title('Dow Jones Index data from 2012 to 2017')
plt.xlabel('Days')
plt.ylabel('Dow Jones close values')
plt.plot(df, label='Per day Close Data')
plt.legend()
plt.show()

# Scale the data using Standard scaler
scaler = StandardScaler()
dow_jones_data_scaled = scaler.fit_transform(df.reshape(-1, 1))
print(dow_jones_data_scaled)

