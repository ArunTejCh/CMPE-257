import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from data import data_preprocessor


df = data_preprocessor.get_cleaned_data("../data/DJI_5_years.csv")
df_train = df[:1000]
df_test = df[1000:]
print(df_train.head())
print(df_test.head())
# df = df.reshape(-1, 1)

# Print shape of dataset
print(df.shape)

# Plot raw data
plt.figure(figsize=(12, 5), frameon=False, facecolor='brown')
plt.title('Dow Jones Index data from 2012 to 2017')
plt.xlabel('Days')
plt.ylabel('Dow Jones close values')
plt.plot(df['Close'], label='Per day Close Data')
plt.legend()
# plt.show()


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
actual_vals = []
for i in range(4, len(df_test)):
    expected = lm.coef_[0] * df_test.iloc[i]['CloseA'] + lm.coef_[1] * df_test.iloc[i]['CloseB'] + lm.coef_[2] * \
                                                                                                   df_test.iloc[i][
                                                                                                       'CloseC'] + \
               lm.coef_[3] * df_test.iloc[i]['CloseD'] + lm.coef_[4] * df_test.iloc[i]['Close'] + lm.intercept_
    print("Predicted: ", expected, "Actual: ", df_test.iloc[i]['Next_Day'])
    predicted_vals.append(expected)
    actual_vals.append(df_test.iloc[i]['Next_Day'])

plt.figure(figsize=(16, 7))
plt.title("Predicted vs Actual values of stocks")
plt.plot(predicted_vals, label="Predicted values")
plt.plot(actual_vals, label="Actual values")
plt.legend()
plt.show()