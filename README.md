# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model

<img width="478" alt="image" src="https://user-images.githubusercontent.com/75235554/187077497-aa09291c-ca42-4031-a487-e2b482a0bf6a.png">

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float','Output':'float'})
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df.head()
x=df[['Input']].values
y=df[['Output']].values
x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=11)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_test1=Scaler.transform(x_test)
x_train1
ai_brain = Sequential([
    Dense(6,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(
    optimizer='rmsprop',
    loss='mse'
)
ai_brain.fit(x_train1,y_train,epochs=4000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
plt.title('Training Loss Vs Iteration Plot')
ai_brain.evaluate(x_test1,y_test)
x_n1=[[66]]
x_n1_1=Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```

## Dataset Information

<img width="344" alt="image" src="https://user-images.githubusercontent.com/75235554/187077589-aad4a2a7-a306-4754-897a-9f25056a9ef2.png">

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="303" alt="image" src="https://user-images.githubusercontent.com/75235554/187077679-94a4cc2c-98c9-407b-8aef-dacb19d81cca.png">

### Test Data Root Mean Squared Error

<img width="412" alt="image" src="https://user-images.githubusercontent.com/75235554/187077878-38ecf52f-43ad-45f9-a27a-717f1b0aeeb4.png">

### New Sample Data Prediction

<img width="211" alt="image" src="https://user-images.githubusercontent.com/75235554/187077752-7c3a455a-ecca-4e5e-87fd-0f8fab77cb36.png">

## RESULT
Succesfully created and trained a neural network regression model for the given dataset.
