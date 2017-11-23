import pandas as pd

dow_jones_data = pd.read_csv("DJI_5_years.csv")
dow_jones_data = dow_jones_data["Close"].values

# Print shape of dataset
print(dow_jones_data)
print(dow_jones_data.shape)

