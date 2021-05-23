import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor as rfr 
from sklearn.model_selection import train_test_split as tts 

file_path="/home/shiva/Documents/healthcare-dataset-stroke-data.csv"
home_data=pd.read_csv(file_path)

l=['age','hypertension','heart_disease','avg_glucose_level','bmi']
X=home_data[l]
y=home_data.stroke
model=rfr(random_state=0)

#calculate mae 
X=X.fillna(X.mean())
y=y.fillna(y.mean())
train_X,val_X,train_y,val_y=tts(X,y,random_state=0)
model.fit(train_X,train_y)
mae=mae(val_y,model.predict(val_X))
model.fit(X,y)

my_data=list(map(float,input("enter the following details\nage hypertension heart disease avg glucose level bmi\n").split()))
my_dataframe=pd.DataFrame([my_data],columns=l)
print("your chance of stroke : ",model.predict(my_dataframe)[0]*100,"% +/- ",mae*100,"%")


