#import Libraries
import numpy as np
import sklearn.datasets

#getting the dataset
breastCancer = sklearn.datasets.load_breast_cancer()

#Separating Data And Label
x= breastCancer.data  # X->Data
y=breastCancer.target # Y-> Target/Label 

#import data to pandas dataframe
import pandas as pd
data = pd.DataFrame(x, columns = breastCancer.feature_names )
data['class'] = y

#count no of overall Cases ClassWise
# print(data['class'].value_counts())


#train & test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state =1) 
#test_size represents % of data to be used for testing and is Optional
# stratify --> for correct distribution of data as of the original data
# random_state --> specific split of data. each value of random_state splits the data differently

#Training Part
#Creating Model and Fitting the data in it
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

#Checking accuracy on training data
from sklearn.metrics import accuracy_score
prediction_on_training_data = model.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy Data : ', accuracy_on_training_data)

# prediction on test_data
prediction_on_test_data = model.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

#test on real data
#MalignInput
# inputData=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189);
#BenignInput
inputData=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
inputDataArray = np.asarray(inputData)
# print(inputDataArray)

# reshape the array as we are predicting the output for one instance
input_data_reshaped = inputDataArray.reshape(1,-1)

#prediction 
prediction = model.predict(input_data_reshaped)
if (prediction[0]==0):
  print('The breast Cancer is Malignant')
else:
  print('The breast cancer is Benign')