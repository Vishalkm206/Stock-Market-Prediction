from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

import streamlit as st


# Title and description
st.title("Stock Price predictor :")


# User input
user_ticker = st.text_input("Enter your Symbol")
# user_ticker = "TATASTEEL.NS"


def load_model(ticker,context_window):
  try:
    return keras.models.load_model('./models/'+str(context_window)+'_model_'+str(ticker)+'.h5')
  except:
    return FileNotFoundError
model = load_model(user_ticker,5)

def predict(model,ticker,context_window):
  df = pd.read_csv('./data/stock_'+str(ticker)+'.csv',header=0,index_col=['Date'])
  x_input = np.array(df['Close'][-(context_window):]).reshape(1,-1)
  temp_input = list(x_input)
  temp_input = temp_input[0].tolist()

  lst_output=[]
  n_steps=context_window
  i=0
  while(i<30):
    if(len(temp_input)>100):
      x_input=np.array(temp_input[1:])
      print("{} day input {}".format(i,x_input))
      x_input=x_input.reshape(1,-1)
      x_input = x_input.reshape((1, n_steps, 1))
      #print(x_input)
      yhat = model.predict(x_input, verbose=0)
      print("{} day output {}".format(i,yhat))
      temp_input.extend(yhat[0].tolist())
      temp_input=temp_input[1:] 
          #print(temp_input)
      lst_output.extend(yhat.tolist())
      i=i+1
    else:
      x_input = x_input.reshape(1, n_steps,1)
      yhat = model.predict(x_input, verbose=0)
      print(yhat[0])
      temp_input.extend(yhat[0].tolist())
      print(len(temp_input))
      lst_output.extend(yhat.tolist())
      i=i+1
  return lst_output

# Get prediction
prediction = predict(model, user_ticker,5)

# Display results
st.write("Your prediction:", prediction)