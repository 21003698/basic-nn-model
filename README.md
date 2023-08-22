# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neurons are the basic input/output units found in neural networks. These units are connected to one another, and each connection carries a weight. Because they are adaptable, neural networks can be applied to both classification and regression. We'll examine how neural networks can be used to tackle regression issues in this post.

A relationship between a dependent variable and one or more independent variables can be established with the aid of regression. Only when the regression equation is a good fit for the data can regression models perform well. Although sophisticated and computationally expensive, neural networks are adaptable and can choose the optimum form of regression dynamically. If that isn't sufficient, hidden layers can be added to enhance prediction. Create your training and test sets using the dataset; in this case, we are creating a neural network with a second hidden layer that uses the activation layer as relu and contains its nodes. We will now fit our dataset before making a value prediction.

## Neural Network Model
![4](https://github.com/21003698/basic-nn-model/assets/93427522/de6a5c18-df45-477c-9c22-78b393f26912)


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

## PROGRAM:
```
## Developed By: Challa Sandeep
## Reg.No:21221240011
```
```
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('data 1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'input':'int','output':'int'})
### Import the packages :
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X = df[['input']].values
Y = df[['output']].values
X

### Split Training and testing set :

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)
X_train1

### Model :

ai_brain = Sequential([
    Dense(5,activation = 'relu'),
    Dense(7,activation = 'relu'),
    Dense(1)])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,Y_train,epochs = 4000)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,Y_test)

X_n1 = [[38]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```

## Dataset Information
![san3](https://github.com/21003698/basic-nn-model/assets/93427522/7e9e4651-60a1-46bd-9aec-5b3715bb75e0)



## OUTPUT

### Training Loss Vs Iteration Plot

![san2](https://github.com/21003698/basic-nn-model/assets/93427522/4c049a47-d835-45db-9d3e-c1e19c3719ec)

### Test Data Root Mean Squared Error
![san2](https://github.com/21003698/basic-nn-model/assets/93427522/674b85dd-7e7d-4210-826c-2de97b498b3b)


### New Sample Data Prediction
![san](https://github.com/21003698/basic-nn-model/assets/93427522/dd3f04a2-c103-4ed3-9493-b2cf5d9657a0)


## RESULT
Therefore a Neural Network Regression Model is developed successfully for the given dataset.
