import matplotlib.pyplot as plt
from data import data_preprocessor
from models import sklearn_linear_regression, tf_ANN
from sklearn.metrics import mean_squared_error
from math import sqrt

df = data_preprocessor.get_cleaned_data("data/DJI_5_years.csv")
df_train = df[:1000]
df_test = df[1000:]

# Plot raw data
plt.figure(figsize=(12, 5), frameon=False, facecolor='brown')
plt.title('Dow Jones Index data from 2012 to 2017')
plt.xlabel('Days')
plt.ylabel('Dow Jones close values')
plt.plot(df['Close'], label='Per day Close Data')
plt.legend()
# plt.show()

linear_predicted_values = sklearn_linear_regression.linear_regression(df_train, df_test)
print(len(linear_predicted_values))
ann_predicted_values = tf_ANN.ann_prediction(df_train, df_test)
print(len(ann_predicted_values))
actual_values = df_test.values[:, 5]
print(len(actual_values))
print("Hi")

# Money Invested
savings_cash = lr_cash = ann_cash = 1000000.0
interest_rate_daily = 0.000052

plt.figure(figsize=(16, 7))
plt.title("Predicted vs Actual values of stocks")
plt.plot(actual_values, label="Actual values")
plt.plot(linear_predicted_values, label="LR Predicted values")
plt.plot(ann_predicted_values, label="ANN Predicted values")
plt.legend()
plt.savefig("comparison.jpg")
# plt.show()

funds_savings_data = []
funds_linear_regression_data = []
funds_ann_data = []
lr_holding_stock = True
ann_holding_stock = True

for i in range(1, len(actual_values)):
    savings_cash = savings_cash + savings_cash * interest_rate_daily
    funds_savings_data.append(savings_cash)
    lr_predicted_stock_price = linear_predicted_values[i]
    if linear_predicted_values[i] > linear_predicted_values[i - 1]:
        lr_holding_stock = True
    else:
        lr_holding_stock = False
    ann_predicted_stock_price = ann_predicted_values[i]
    if ann_predicted_values[i] > ann_predicted_values[i - 1]:
        ann_holding_stock = True
    else:
        ann_holding_stock = False
    if lr_holding_stock:
        lr_cash = lr_cash * (actual_values[i] / actual_values[i - 1])
    if ann_holding_stock:
        ann_cash = ann_cash * (actual_values[i] / actual_values[i - 1])
    funds_linear_regression_data.append(lr_cash)
    funds_ann_data.append(ann_cash)

plt.figure(figsize=(16, 7))
plt.title("Profit Comparison, Saving account vs ML models")
plt.plot(funds_savings_data, label="Savings funds")
plt.plot(funds_linear_regression_data, label="LR funds")
plt.plot(funds_ann_data, label="ANN funds")
plt.legend()
plt.savefig("funds_comparison.jpg")
plt.show()

rms_LR = sqrt(mean_squared_error(actual_values, linear_predicted_values))
rms_ANN = sqrt(mean_squared_error(actual_values, ann_predicted_values))

print("LR error: ", rms_LR, "ANN error: ", rms_ANN)

print("Savings profit $", round(savings_cash - 1000000.0, 2), " Interest rate", 1.30, "% LR profit $", round(lr_cash - 1000000.0, 2),
      " Interest rate", round(((lr_cash - 1000000.0)/1000000) * 100.0, 2), "%  ANN profit $",
      round(ann_cash - 1000000.0, 2), " Interest rate ", round(((ann_cash - 1000000.0)/1000000) * 100.0, 2), "%")
