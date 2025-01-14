import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go

start = '2010-01-01'
end = '2025-12-31'
st.title('Stock Treand Prediction')
user_input=st.text_input('Enter Stock Ticker','TCS') 
df = yf.download(user_input, start=start, end=end)

# describe the data
st.subheader('Data of the Stock')
st.write(df.describe()) 

# visuallizing
st.subheader('closing price Vs Time chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
st.subheader('Closing Price Vs Time chart with 100MA & 200MA')

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig, ax = plt.subplots(figsize=(12, 6))

# Specify colors for each line
ax.plot(ma100, label='MA100', color='red')
ax.plot(ma200, label='MA200', color='blue')
ax.plot(df.Close, label='Closing Price', color='black')
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

#spliting data into training and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])  

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#load my modle
model=load_model('stock_model.h5')

#testing part
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range (100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted= model.predict(x_test)

#scaleing the values
scaler = scaler.scale_
scale_factor=1/scaler[0]
y_predicted =y_predicted*scale_factor
y_test =y_test*scale_factor

#final graph
st.subheader('predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Price')
plt.ylabel('True')
plt.legend()
st.pyplot(fig2)


