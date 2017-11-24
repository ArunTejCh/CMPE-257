import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression


df = pd.read_csv("DJI_5_years.csv")
df['Next_Day'] = pd.Series(df['Close'], index=df.index)
df['CloseA'] = pd.Series(df['Close'], index=df.index)
df['CloseB'] = pd.Series(df['Close'], index=df.index)
df['CloseC'] = pd.Series(df['Close'], index=df.index)
df['CloseD'] = pd.Series(df['Close'], index=df.index)
print(df.head())
for i in range(4, len(df) - 1):
    df.at[i, 'Next_Day'] = df.iloc[i+1]['Close']
    df.at[i, 'CloseA'] = df.iloc[i - 1]['Close']
    df.at[i, 'CloseB'] = df.iloc[i - 2]['Close']
    df.at[i, 'CloseC'] = df.iloc[i - 3]['Close']
    df.at[i, 'CloseD'] = df.iloc[i - 4]['Close']

df = df[["CloseA", "CloseB", "CloseC", "CloseD", "Close", "Next_Day"]].copy()
df = df.iloc[4:]

df_train = df[:1000]
df_test = df[1000:]
print(df_train.head())
print(df_test.head())
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
#plt.show()

# create a fitted model with all three features
lm = smf.ols(formula='Next_Day ~ CloseA + CloseB + CloseC + CloseD + Close', data=df_train).fit()

# print the coefficients
print(lm.params)


feature_cols = ['CloseA', 'CloseB', 'CloseC', 'CloseD', 'Close']
X = df_train[feature_cols]
y = df_train.Next_Day

# Linear regression using scikit-learn
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


print(df_test.shape)
predicted_vals = []
actual_vals= []
for i in range(4, len(df_test)):
    expected = lm.coef_[0] * df_test.iloc[i]['CloseA'] + lm.coef_[1] * df_test.iloc[i]['CloseB'] + lm.coef_[2] * df_test.iloc[i]['CloseC'] + lm.coef_[3] * df_test.iloc[i]['CloseD'] + lm.coef_[4] * df_test.iloc[i]['Close'] + lm.intercept_
    print("Predicted: ",expected, "Actual: ", df_test.iloc[i]['Next_Day'])
    predicted_vals.append(expected)
    actual_vals.append(df_test.iloc[i]['Next_Day'])

plt.figure(figsize=(16, 7))
plt.title("Predicted vs Actual values of stocks")
plt.plot(predicted_vals, label="Predicted values")
plt.plot(actual_vals, label="Actual values")
plt.legend()
plt.show()