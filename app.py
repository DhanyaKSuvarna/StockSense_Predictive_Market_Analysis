from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.title('StockSense: Predictive Market Analysis')

user_input = st.text_input('Enter Stock Ticker', 'AAPL').upper()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Select Start Date", date(2015, 1, 1))
with col2:
    end_date = st.date_input("Select End Date", date(2024, 12, 31))

owns_stock = st.checkbox("I already own this stock", value=False)

df = yf.download(user_input, start=start_date, end=end_date)

if df.empty:
    st.error("❌ No data found for this stock/date range.")
    st.stop()

df.reset_index(inplace=True)

if 'Adj Close' in df.columns:
    df.drop(['Adj Close'], axis=1, inplace=True)

st.write("### Stock Data for:", user_input)
st.dataframe(df)


# Close Price
st.subheader('Close Price vs Time Chart')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], label='Close Price')
ax.set_title("Close Price vs Time")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# MA100
st.subheader('Close Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], label='Close Price')
ax.plot(ma100, 'r', label='MA100')
ax.set_title("Close Price with 100-Day Moving Average")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# MA200
st.subheader('Close Price vs Time Chart with 200MA')
ma200 = df['Close'].rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Close'], label='Close Price')
ax.plot(ma200, 'g', label='MA200')
ax.set_title("Close Price with 200-Day Moving Average")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

training_size = int(len(df) * 0.70)
data_training = df['Close'][:training_size]
data_testing = df['Close'][training_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

model = load_model("C:\\Users\\dhany\\OneDrive\\Documents\\DSML_Project\\keras_model_universal.h5", compile=False)

# Prepare test data
past_100 = data_training.tail(100)
final_df = pd.concat([past_100, data_testing], axis=0)

scaled_final = scaler.transform(np.array(final_df).reshape(-1, 1))

x_test, y_test = [], []
for i in range(100, len(scaled_final)):
    x_test.append(scaled_final[i-100:i])
    y_test.append(scaled_final[i])

x_test, y_test = np.array(x_test), np.array(y_test)

# Model prediction
predicted = model.predict(x_test)

# Reverse scaling properly
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test)

from sklearn.metrics import mean_squared_error

# Compute RMSE
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# Convert RMSE to percentage accuracy
# Formula: accuracy = 100 - (RMSE / mean_actual_price * 100)
mean_actual = np.mean(actual_prices)
accuracy = max(0, 100 - (rmse / mean_actual * 100))

st.subheader("🎯 Model Accuracy")

st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**Estimated Accuracy:** {accuracy:.2f}%")


st.subheader('Predictions vs Original')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(actual_prices, label='Actual Price')
ax.plot(predicted_prices, label='Predicted Price')
ax.set_title("Predictions vs Actual Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)


last_close = actual_prices[-1][0]
next_predicted_price = predicted_prices[-1][0]

change_percent = ((next_predicted_price - last_close) / last_close) * 100

st.subheader("📌 AI Recommendation")

st.write(f"*Last Closing Price:* ${last_close:.2f}")
st.write(f"*Next Predicted Price:* ${next_predicted_price:.2f}")
st.write(f"*Expected Change:* {change_percent:.2f}%")

threshold = 2  # ±2% change threshold

if change_percent > threshold:
    signal = "BUY"
elif change_percent < -threshold:
    signal = "SELL"
else:
    signal = "HOLD"

# If user owns stock → HOLD instead of BUY
if owns_stock and signal == "BUY":
    signal = "HOLD"

st.write(f"### 📢 *Final Signal: {signal}*")

if signal == "BUY":
    st.success("The model expects the stock to rise. Consider BUYING.")
elif signal == "SELL":
    st.error("The model expects the stock to fall. Consider SELLING.")
else:
    st.info("Price expected to stay stable. Consider HOLDING.")