import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/Metro_Interstate_Traffic_Volume.csv")

# Peek at data
print(data.head())

# Basic Cleaning
data = data.dropna(subset=["date_time", "traffic_volume"]) # remove missing values

#Feature Engineering
data["datetime"] = pd.to_datetime(data["date_time"])
data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek

# Group by hour and calculate mean traffic volume
hourly_traffic = data.groupby("hour")["traffic_volume"].mean()

# Plot hourly traffic pattern
plt.figure(figsize=(10,5))
hourly_traffic.plot()
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Traffic Volume")
plt.grid()
plt.show()

# Heatmap of traffic by day & hour
pivot = data.pivot_table(values="traffic_volume", index="day_of_week", columns="hour", aggfunc=np.mean)
plt.imshow(pivot, cmap="viridis")
plt.colorbar(label="Traffic Volume")
plt.title("Traffic Heatmap (Day vs Hour)")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.show()

#GroupBy weather to identify the traffic volume
weather = [c for c in dict.fromkeys(data["weather_main"])]
weather_wrt_traffic = data.groupby("weather_main")["traffic_volume"].mean()
cmap = plt.get_cmap("tab10", len(weather))
color = [cmap(i) for i in range(len(weather))]
plt.figure(figsize=(11,6))
weather_wrt_traffic.plot(kind="bar", color=color)
plt.title("Average Traffic Volume by Weather")
plt.xlabel("Weather")
plt.ylabel("Traffic Volume")
plt.grid()
plt.show()