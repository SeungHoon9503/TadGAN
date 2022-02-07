#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#데이터 가져오기
data=pd.read_excel("B기업_데이터 (비식별화).xlsx")

#1. 데이터 제거
#'운전 누적 시간'이 없는 데이터 제거 - '현재상태'=='운전중'
index=data[data['현재 상태']=='운전중'].index
data=data.drop(index)


#맨 밑단에 NAN제거
data=data.iloc[:-2]

#중간에 NAN이 많은 데이터 한 줄 삭제
index=data[data['No.']==566].index
data=data.drop(index)



#2. 중복 feature 제거
data=data.drop(['1-수소공급량.1','2-수소공급량.1', '1-산소공급량.1', 
                '2-산소공급량.1', '1-버너회전수.1', '2-버너회전수.1', 
                '1-버너속도.1', '2-버너속도.1', '1-핀치횟수.1', '2-핀치횟수.1', 
                '핀치두께-상.1', '핀치두께-중.1', '핀치두께-하.1', 
                '핀치두께2-상.1', '핀치두께2-중.1', '핀치두께2-하.1', 
                '전극간격.1', '채널.1', 'VERSION.1', '전압(1차).1', 
                '전류(1차).1', '역률(1차).1', '전력(1차).1', '온도(1차).1', 
                '전압(2차).1', '전류(2차).1', '역률(2차).1', '전력(2차).1', 
                '온도(2차).1', '강도.1'],axis=1)
#print(list(data.columns))



#3. 필요없는 feature 삭제
data=data.drop(['No.', 'Test No.', 'Lamp Ver. No.', 'Chamber No.', 'Lamp No.', 
                'Lot No.','VERSION','운전 시작일', '운전 종료일','불량코드', '판정', 
                '측정시간', '합불여부', '등록자', '등록일시', 
                '수정자', '수정일시', 'Head No.', '1-1\n(Pinch No.1, U돌출)', 
                '1-2\n(Pinch No.1, U평탄)', '2-1\n(Pinch No.2, U돌출)', 
                '2-2\n(Pinch No.2, U평탄)', '1-1\n(Pinch No.1, 배꼽 위로)', 
                '1-2\n(Pinch No.1, 배꼽 아래)', '2-1\n(Pinch No.2, 배꼽 위로)', 
                '2-2\n(Pinch No.2, 배꼽 아래)','Leak No.', '리크 부위', 
                'Pinch No. 1\n(℃)', '중앙\n(℃)', 'Pinch No. 2\n(℃)'],axis=1)


#4. 현재상태 범주화
data['현재 상태']=data['현재 상태'].replace('Alive',0)
data['현재 상태']=data['현재 상태'].replace('Gas leak',1)
data['현재 상태']=data['현재 상태'].replace('Lamp fail',2)

#5. 결측치 대체
#온도(1차)
medi=data['온도(1차)'].median()
data['온도(1차)']=data['온도(1차)'].replace('NAN',medi)
#전압(2차)
medi=data['전압(2차)'].median()
data['전압(2차)']=data['전압(2차)'].replace('NAN',medi)
#역률(2차)
medi=data['역률(2차)'].median()
data['역률(2차)']=data['역률(2차)'].replace('NAN',medi)
#전력(2차)
medi=data['전력(2차)'].median()
data['전력(2차)']=data['전력(2차)'].replace('NAN',medi)
#온도(2차)
medi=data['온도(2차)'].median()
data['온도(2차)']=data['온도(2차)'].replace('NAN',medi)



#6. 운전누적시간
df=data['운전 누적 시간\n(On-Off 횟수)']
drivingTime=[]

for i in df:
    time_OnOff=i.split("(")
    drivingTime.append(time_OnOff[0][:-1])

for i,x in enumerate(drivingTime):
    h=x.split(":")[0]
    m=x.split(":")[1]
    minute=int(h)*60+int(m)
    drivingTime[i]=minute
  
df=pd.DataFrame(df)
df['운전 누적 시간\n(On-Off 횟수)']=drivingTime
data['운전 누적 시간\n(On-Off 횟수)']=df
data.rename(columns={'운전 누적 시간\n(On-Off 횟수)':'운전 누적 시간'},inplace=True)

#7. 이상치 제거
def get_outlier(df=None, column=None, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    quantile_25 = np.percentile(df[column].values, 25)
    quantile_75 = np.percentile(df[column].values, 75)
    
    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight
    
    lowest = quantile_25-IQR_weight
    highest = quantile_75 + IQR_weight
    
    outlier_idx = df[column][(df[column]<lowest)|(df[column]>highest)].index
    return outlier_idx

outlier_idx=get_outlier(data, '운전 누적 시간')
data.drop(outlier_idx, axis=0, inplace=True)
outlier_idx=get_outlier(data, '운전 누적 시간')
data.drop(outlier_idx, axis=0, inplace=True)

data.to_csv("해대후.csv",mode='w',encoding='utf-8-sig')


# In[2]:


###Shap value feature selection###
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#전처리된 데이터 불러오기
data=pd.read_csv("해대후.csv",index_col=0)
data.rename(columns={'Pinch No.1\n정면부 측정 편차':'Pinch No.1(정면부 측정 편차)',
                    'Pinch No.1\n측면부 측정 편차':'Pinch No.1(측면부 측정 편차)',
                    'Pinch No.2\n정면부 측정 편차':'Pinch No.2(정면부 측정 편차)',
                    'Pinch No.2\n측면부 측정 편차':'Pinch No.2(측면부 측정 편차)'},inplace=True)

#Scaling
scaler=MinMaxScaler()
data[:]=scaler.fit_transform(data[:])
X=data.drop(['운전 누적 시간'], axis=1)
Y=data['운전 누적 시간']

#train:test = 7:3
X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.3, shuffle=True, random_state=42)


import lightgbm as lgb
from math import sqrt
from sklearn.metrics import mean_squared_error

#lightgbm 학습
lgb_dtrain=lgb.Dataset(X_train,y_train)
lgb_param = {'max_depth': 10,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'objective': 'regression'}
lgb_model = lgb.train(params = lgb_param, train_set = lgb_dtrain)
lgb_model_predict = lgb_model.predict(X_test)

#한글 깨짐 방지
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

set(sorted([f.name for f in mpl.font_manager.fontManager.ttflist]))

mpl.rc('font', family='Batang')

import shap
explainer = shap.TreeExplainer(lgb_model) # Tree model Shap Value 확인 객체 지정
shap_values = explainer.shap_values(X_test) # Shap Values 계산
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#전처리된 데이터 불러오기
data=pd.read_csv("해대후.csv",index_col=0)

'''
#tree기반 feature
data=data[['역률(2차)', '전력(1차)', '전압(1차)', '핀치두께2-하', '핀치두께2-중', '1-핀치횟수',
          '현재 상태', '운전 누적 시간','강도']]
'''

#shap value기반 feature
data=data[['현재 상태', '2-수소공급량','핀치두께2-하',
           '역률(2차)','전류(1차)','핀치두께-하',
           '핀치두께-상','운전 누적 시간','강도']]

#Scaling
scaler=MinMaxScaler()
data[:]=scaler.fit_transform(data[:])
X=data.drop(['운전 누적 시간'], axis=1)
Y=data['운전 누적 시간']



#train:test = 7:3
X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.3, shuffle=True, random_state=42)


# In[4]:


#SVR
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#SVR model
model=SVR()

#model training
model.fit(X_train, y_train)

#prediction
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x,y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[5]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

#RandomForestRegressor model
model=RandomForestRegressor()

#model training
model.fit(X_train, y_train)

#prediction
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x, y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[6]:


#XGBoostRegressor
from xgboost import XGBRegressor

#XGBoostRegressor model
model=XGBRegressor()

#model training
model.fit(X_train, y_train)

#predictin
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[7]:


#DNN
import tensorflow.keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

model = Sequential()

# input layer, first hidden layer
model.add(Dense(units=512, kernel_initializer='he_normal', activation='relu', input_dim=8))

# hidden layer
model.add(Dense(units=256, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=128, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=64, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=16, kernel_initializer='he_normal', activation='relu'))

# output layer
model.add(Dense(units=1))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#compiling the ANN
model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])

#fitting the ANN to the Training set
from sklearn.metrics import r2_score

model.fit(X_train, y_train, epochs = 300)

#predictin
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[8]:


#Decision Tree Feature Selection
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#전처리된 데이터 불러오기
data=pd.read_csv("해대후.csv",index_col=0)

#종속변수 범주화
bins = [0, 30000, 60000]
data['g_Time'] = np.digitize(data['운전 누적 시간'], bins)
feature = data.drop(['운전 누적 시간', 'g_Time'], axis=1)
target = data['g_Time']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.3, random_state = 0)

dt_clf = DecisionTreeClassifier(random_state=0, max_depth=7)
dt_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = dt_clf.predict(x_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

feature_name = feature.columns

from sklearn.tree import export_graphviz

export_graphviz(
    dt_clf, # 학습한 모형
    out_file = './tree.dot', # .dot 파일 저장 위치
    feature_names = feature_name, # 사용한 변수 이름
    class_names = ['1','2','3'],# 예측할 타겟 클래스 이름
    rounded = True, # 사각형 끝을 둥글게
    filled=True # 사각형 안 색깔 채우기
)

import graphviz
# 위에서 생성된 tree.dot 파일을 Graphiviz 가 읽어서 시각화
with open("tree.dot", encoding='UTF-8') as f:
    dot_graph = f.read()
src = graphviz.Source(dot_graph)

#한글깨짐 방지 코드 
import matplotlib
# from matplotlib import font_manager, rc
# import platform
# if platform.system()=="Windows":
#     font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#     rc('font', family=font_name)

import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic') 
    

def plot_feature_importances_feature(model):
    plt.figure(figsize=(15,5))
    n_features = x_test.shape[1]
    plt.barh(np.arange(n_features), dt_clf.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), x_test.columns)
    plt.xticks()
    plt.xlabel("특성 중요도")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
    
plot_feature_importances_feature(dt_clf)


# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#전처리된 데이터 불러오기
data=pd.read_csv("해대후.csv",index_col=0)

'''
#shap value기반 feature
data=data[['현재 상태', '2-수소공급량','핀치두께2-하',
           '역률(2차)','전류(1차)','핀치두께-하',
           '핀치두께-상','운전 누적 시간','강도']]
'''

#tree기반 feature
data=data[['역률(2차)', '전력(1차)', '전압(1차)', '핀치두께2-하', '핀치두께2-중', '1-핀치횟수',
          '현재 상태', '운전 누적 시간','강도']]

#Scaling
scaler=MinMaxScaler()
data[:]=scaler.fit_transform(data[:])
X=data.drop(['운전 누적 시간'], axis=1)
Y=data['운전 누적 시간']



#train:test = 7:3
X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.3, shuffle=True, random_state=42)


# In[10]:


#SVR
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#SVR model
model=SVR()

#model training
model.fit(X_train, y_train)

#prediction
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x,y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[11]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

#RandomForestRegressor model
model=RandomForestRegressor()

#model training
model.fit(X_train, y_train)

#prediction
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x, y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[12]:


#XGBoostRegressor
from xgboost import XGBRegressor

#XGBoostRegressor model
model=XGBRegressor()

#model training
model.fit(X_train, y_train)

#predictin
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[13]:


#DNN
import tensorflow.keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

model = Sequential()

# input layer, first hidden layer
model.add(Dense(units=512, kernel_initializer='he_normal', activation='relu', input_dim=8))

# hidden layer
model.add(Dense(units=256, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=128, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=64, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=32, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(units=16, kernel_initializer='he_normal', activation='relu'))

# output layer
model.add(Dense(units=1))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#compiling the ANN
model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])

#fitting the ANN to the Training set
from sklearn.metrics import r2_score

model.fit(X_train, y_train, epochs = 300)

#predictin
x=np.array([i for i in range(len(X_test))])
y_p=model.predict(X_test)

plt.scatter(x,y_test, marker='+')
plt.scatter(x, y_p, marker='o')
plt.show()

#RMSE, MAPE
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
from math import sqrt

mse=mean_squared_error(y_test, y_p)
print("RMSE: ",sqrt(mse))
print("MAPE", mean_absolute_percentage_error(y_test,y_p))


# In[ ]:




