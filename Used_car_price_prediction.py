#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Importing the required models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# importing the regression metrics
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error





# In[2]:


df_train = pd.read_csv('used_cars_train-data.csv')
df_test = pd.read_csv('used_cars_test-data.csv')
df_train.head()


# In[3]:


df_train.shape


# In[4]:


# 'New_price' attribute has more than 80% of null values. Hence I will drop that column
df_train.isnull().sum()/len(df_train)


# In[5]:


# Dropping the 'New_price' and 'Unnamed: 0' column
df_train = df_train.drop(['New_Price','Unnamed: 0'],axis=1)
df_test = df_test.drop(['New_Price','Unnamed: 0'],axis=1)


# In[6]:


df_train.isnull().sum()/len(df_train)


# In[7]:


df_train.dtypes


# Note: Out of ['Mileage','Engine','Power','Seats'] features, ['Mileage','Engine','Power'] are object datatypes. We can convert them
# into numerical values

# In[8]:


(df_train['Mileage']).str.split().str[1].unique()


# In[9]:


# Splitting the Name column into Brand_name and car_name
df_train['Brand_name'] = df_train['Name'].str.split().str[0]
df_test['Brand_name'] = df_test['Name'].str.split().str[0]


# In[10]:


# Extracting the carname from 'Name' feature
def extract_carname(df):
    car_name=[]
    for car in df['Name'].str.split().str[1:]:
        car_name.append(' '.join(car))
    df['Car_name'] = pd.Series(car_name)
    return df


# In[11]:


df_train = extract_carname(df_train)
df_test = extract_carname(df_test)


# In[12]:


df_train = df_train.replace({'Power':{'null bhp':'0',np.nan:'0'}})
df_test = df_test.replace({'Power':{'null bhp':'0',np.nan:'0'}})


# In[13]:


# Replacing the strings values in 'Mileage', 'Engine' and 'Power'
# 1 kg LPG gas is 1.96 liters so 1km/kg = 1km/1.96l => 0.510 kmpl
df_train['Mileage(kmpl)'] = np.where(df_train['Mileage'].str.split().str[1]=='km/kg', df_train['Mileage'].str.split().str[0].astype('float32')*(0.510),df_train['Mileage'].str.split().str[0] )

df_test['Mileage(kmpl)'] = np.where(df_test['Mileage'].str.split().str[1]=='km/kg', df_test['Mileage'].str.split().str[0].astype('float32')*(0.510),df_test['Mileage'].str.split().str[0])

df_train['Mileage(kmpl)'].astype('float32')
df_test['Mileage(kmpl)'].astype('float32')

# Extracting only the numerical portion of the features for 'Engine' and 'Power'
df_train['Engine(CC)'] = df_train['Engine'].str.split().str[0].astype('float32')
df_test['Engine(CC)'] = df_test['Engine'].str.split().str[0].astype('float32')

df_train['Power(bhp)'] = df_train['Power'].str.replace(' bhp','',regex=True)
df_test['Power(bhp)'] = df_test['Power'].str.replace(' bhp','',regex=True)


# In[14]:


# Removing the unwanted columns
df_train = df_train.drop(['Mileage','Engine','Power'], axis=1)
df_test = df_test.drop(['Mileage','Engine','Power'], axis=1)


# In[15]:


df_train.isnull().sum() # ['Seats','Mileeage(kmpl)','Engine(CC)','Power(bhp)']


# In[16]:


# Getting thte mean value of seats based on different brand names
seat_fill = round(df_train[['Brand_name','Seats']].groupby('Brand_name').mean(),0)
seat_fill_test = round(df_test[['Brand_name','Seats']].groupby('Brand_name').mean(),0)


# In[17]:


# Merging the dataframes to replace the nan values of seats 
merge_df = pd.merge(df_train,seat_fill,how='left',left_on=df_train['Brand_name'],right_on=seat_fill.index)
merge_dft = pd.merge(df_test,seat_fill_test,how='left',left_on=df_test['Brand_name'],right_on=seat_fill_test.index)


# In[18]:


# Filling the na values of seats with mean values found
merge_df['Seats_x'].fillna(merge_df['Seats_y'],inplace=True)
merge_dft['Seats_x'].fillna(merge_dft['Seats_y'],inplace=True)


# In[19]:


# Dropping the irrelavant columns
merge_df.drop(['key_0','Seats_y'],axis=1,inplace=True)
merge_dft.drop(['key_0','Seats_y'],axis=1,inplace=True)


# In[20]:


# Dropping the Name column as it's already split into brand and car_name
merge_df.drop(['Name'],axis=1,inplace=True)
merge_dft.drop(['Name'],axis=1,inplace=True)


# In[21]:


# Dropping the remaining Nan columns
merge_df.dropna(inplace=True)
merge_dft.dropna(inplace=True)


# In[22]:


merge_dft.isnull().sum()


# In[23]:


merge_df.head()


# In[24]:


merge_df.dtypes[merge_df.dtypes=='object']


# In[25]:


# Converting Mileage(kmpl) feature into float32
merge_df['Mileage(kmpl)'] = merge_df['Mileage(kmpl)'].astype('float32')
merge_dft['Mileage(kmpl)'] = merge_dft['Mileage(kmpl)'].astype('float32')

merge_df['Power(bhp)'] = merge_df['Power(bhp)'].astype('float32')
merge_dft['Power(bhp)'] = merge_dft['Power(bhp)'].astype('float32')


# In[26]:


merge_df.dtypes[merge_df.dtypes=='object']


# In[27]:


merge_df.head()


# In[28]:


merge_df.reset_index(drop=True,inplace=True)
merge_dft.reset_index(drop=True,inplace=True)


# In[29]:


def extract_cars(df):
    cars = []
    for car in df['Car_name'].str.split():
        cars.append(' '.join(car[:2]))
#         if len(car[0])<=2:
#             cars.append(' '.join(car[:2]))
#          else:
#             cars.append(car[0])
    s = pd.Series(cars,name='Cars')
    return s


# In[30]:


# Extracting the car_name alone from the car_names feature
merge_df['Cars'] = extract_cars(merge_df)
merge_dft['Cars'] = extract_cars(merge_dft)


# In[31]:


# Dropping the Car_name feature
merge_df.drop('Car_name', axis=1, inplace=True)
merge_dft.drop('Car_name', axis=1, inplace=True)


# In[32]:


merge_df.head()


# In[33]:


# Analysing the object columns
for col in ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand_name','Cars']:
    print(f"{col}: ",list(merge_df[col].unique()),"\n")


# In[34]:


print(merge_df.columns)
print(merge_dft.columns)


# In[35]:


# Splitting input and output & train and test data

X = merge_df.drop('Price',axis=1).copy()
y = merge_df['Price'].copy()

Xt = merge_dft.copy()


# In[36]:


X.head()


# In[37]:


# Changing the order of the columns
cols = ['Brand_name', 'Transmission', 'Fuel_Type', 'Owner_Type', 'Location', 'Cars',         'Year', 'Kilometers_Driven', 'Mileage(kmpl)', 'Engine(CC)', 'Power(bhp)', 'Seats_x']
X = X[cols]
Xt = Xt[cols]


# In[38]:


X.head()


# In[39]:


# Changing the column names before splitting the data 
#(note: columnTransformer gives keyerror if fitted using column names and dont' use the column name later)

X.rename({'Brand_name': 0, 'Transmission': 1, 'Fuel_Type': 2, 'Owner_Type': 3,           'Location': 4, 'Cars': 5, 'Year': 6, 'Kilometers_Driven': 7, 'Mileage(kmpl)': 8,           'Engine(CC)': 9, 'Power(bhp)': 10, 'Seats_x': 11},axis=1, inplace=True)
Xt.rename({'Brand_name': 0, 'Transmission': 1, 'Fuel_Type': 2, 'Owner_Type': 3,           'Location': 4, 'Cars': 5, 'Year': 6, 'Kilometers_Driven': 7, 'Mileage(kmpl)': 8,           'Engine(CC)': 9, 'Power(bhp)': 10, 'Seats_x': 11}, axis=1, inplace=True)


# In[40]:


X.columns, Xt.columns


# In[44]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=41)


# In[45]:


# OneHotEncoding categorical columns except 'Car_name'

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder

# Specify the column names supported only inside the pandas dataframes when deployed it asks for position of the column
# ct_transform = ColumnTransformer([('encoder1', OneHotEncoder(handle_unknown='ignore'), \
#                                    ['Brand_name','Transmission','Fuel_Type','Owner_Type','Location']),\
#                                    ('encoder2', BinaryEncoder(), ['Cars'])],remainder='passthrough')

ct_transform = ColumnTransformer([('encoder1', OneHotEncoder(handle_unknown='ignore'),                                    [0,1,2,3,4]),                                   ('encoder2', BinaryEncoder(), [5])],remainder='passthrough')


# In[47]:


transformed_xtrain = ct_transform.fit_transform(X_train)


# In[147]:


Xttrain = pd.DataFrame(ct_transform.fit_transform(X_train),columns=ct_transform.get_feature_names())
Xttest = pd.DataFrame(ct_transform.transform(X_test),columns=ct_transform.get_feature_names())
dftest  = pd.DataFrame(ct_transform.transform(Xt),columns=ct_transform.get_feature_names())
Xttrain.head()


# In[148]:


pd.set_option('display.max_columns',None)


# In[149]:


list(Xttrain.columns)


# In[150]:


Xttrain.shape


# In[151]:


# Exporting the transformer as pickle file
pickle.dump(ct_transform,open('column_transformer.pkl','wb'))


# In[48]:


# Instantiating the models
lr_model = LinearRegression(normalize=True)
dt_model = DecisionTreeRegressor(min_samples_split=5)
sv_model = SVR()
rf_model = RandomForestRegressor(n_estimators = 300, min_samples_split=5)
knn = KNeighborsRegressor(n_neighbors=5)


# ### 1. Using Linear Regression model

# In[153]:


X = Xttrain.copy()
y = y_train.copy()
Xt = Xttest.copy()
yt = y_test.copy()


# In[154]:


score_board=dict()


# In[155]:


# Linear Regression model evaluation
lr_model.fit(X,y)
y_hat_lr = lr_model.predict(Xt)

score_board['Linear_Regression']=[np.sqrt(mean_squared_error(yt,y_hat_lr)),r2_score(yt,y_hat_lr)]

print(np.sqrt(mean_squared_error(yt,y_hat_lr)))
print(r2_score(yt,y_hat_lr))


# In[156]:


plt.figure(figsize=(10,6))
sns.distplot(yt,color='green',hist=False)
sns.distplot(y_hat_lr,color='blue', hist=False)
plt.show()


# ### 2. Using Decision Tree model

# In[157]:


dt_model.fit(X,y)
y_hat_dt = dt_model.predict(Xt)

score_board['Decition_tree']=[np.sqrt(mean_squared_error(yt,y_hat_dt)),r2_score(yt,y_hat_dt)]

print('RMSE:', np.sqrt(mean_squared_error(yt,y_hat_dt)))
print('R2_score:', r2_score(yt,y_hat_dt))


# In[158]:


plt.figure(figsize=(10,6))
sns.distplot(yt,color='green',hist=False)
sns.distplot(y_hat_dt,color='blue',hist=False)
plt.show()


# ### 3. Using Support Vector machine model

# In[159]:


sv_model.fit(X,y)
y_hat_sv = sv_model.predict(Xt)

score_board['SVM']=[np.sqrt(mean_squared_error(yt,y_hat_sv)),r2_score(yt,y_hat_sv)]

print('RMSE:', np.sqrt(mean_squared_error(yt,y_hat_sv)))
print('R2_score:', r2_score(yt,y_hat_sv))


# In[160]:


plt.figure(figsize=(10,6))
sns.distplot(yt,color='green',hist=False)
sns.distplot(y_hat_sv,color='blue',hist=False)
plt.show()


# ### 4. Using Random Forest model

# In[161]:


rf_model.fit(X,y)
y_hat_rf = rf_model.predict(Xt)

score_board['Random_Forest']=[np.sqrt(mean_squared_error(yt,y_hat_rf)),r2_score(yt,y_hat_rf)]

print('RMSE:', np.sqrt(mean_squared_error(yt,y_hat_rf)))
print('R2_score:', r2_score(yt,y_hat_rf))


# In[162]:


plt.figure(figsize=(10,6))
sns.distplot(yt,color='green',hist=False)
sns.distplot(y_hat_rf,color='blue',hist=False)
plt.show()


# ### 5. Using KNN model

# In[163]:


knn.fit(X,y)
y_hat_knn = knn.predict(Xt)

score_board['KNN']=[np.sqrt(mean_squared_error(yt,y_hat_knn)),r2_score(yt,y_hat_knn)]

print('RMSE:', np.sqrt(mean_squared_error(yt,y_hat_knn)))
print('R2_score:', r2_score(yt,y_hat_knn))


# In[164]:


plt.figure(figsize=(10,6))
sns.distplot(yt,color='green',hist=False)
sns.distplot(y_hat_knn,color='blue',hist=False)
plt.show()


# In[165]:


scores = pd.DataFrame(score_board,index=['RMSE','R2_score']).T
scores.sort_values('RMSE',ascending=True)


# ### Note: From the above scores, we can understand that Random Forest is the best predictor

# In[166]:


# Exporting the random forest model
pickle.dump(rf_model,open('Rf_model.pkl','wb'))


# In[169]:


# Testing with sample data
#column order in which input values are provided
# ['Brand_name','Transmission','Fuel_Type','Owner_Type','Location','Cars','Year','Kilometers_Drivern','Mileage','Engine'\
# ,'Power','seat']

sample_input =  ['Ambassador', 'Manual', 'Diesel', 'two', 'Hyderabad', 'Classic Nova Diesel', '2019',                  '500', '12', '5900', '400', '3']


# In[167]:


# loading the transformers and trained model
transformer = pickle.load(open('column_transformer.pkl','rb'))
model = pickle.load(open('rf_model.pkl','rb'))


# In[173]:


# input sample
print(np.array([sample_input]))

# Converting it to dataframe for column Transformer to transform the columns
pd.DataFrame(np.array([sample_input]))


# In[ ]:




