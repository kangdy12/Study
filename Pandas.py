#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


titanic_df = pd.read_csv(r'titanic_train.csv')
titanic_df.head(3)


# In[3]:


print('titanic 변수 타입 : ', type(titanic_df))
titanic_df.shape
titanic_df.info()


# - 머신러닝 알고리즘 : 데이터의 분포도를 아는 것이 알고리즘의 성능을 향상시키는 중요한 요소
# 
# ex) 회귀 : 데이터가 정규 분포를 이루지 않고 특정 값으로 왜곡돼 있는 경우 or 이상치가 많으면 예측성능 저하

# In[4]:


titanic_df.describe() # R : summary랑 비슷


# - value_counts() : 데이터 분포도 확인하는데 유용

# In[5]:


value_counts = titanic_df['Pclass'].value_counts()
value_counts


# ### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환

# In[6]:


import numpy as np
col_name1 = ['col1']
list1 = [1,2,3]
array1 = np.array(list1)

##dimension is 1
#리스트를 이용해 DataFrame 생성
df_list1 = pd.DataFrame(list1, columns = col_name1)
df_list1


# In[7]:


##2x3
col_name2 = ['col1','col2','col3']
list2 = [[11,12,13],
        [1,2,3]]
df2 = pd.DataFrame(list2,columns = col_name2)
df2

#dictionary로
dict1 = {'col1' : [11,1], 'col2' : [12,2], 'col3' : [13,3]}
df3 = pd.DataFrame(dict1)
df3


# - df3(dict로 데이터프레임 만든 것) >> ndarray로 변환

# In[8]:


array3= df3.values # << ndarray
print(array3)
df3.values.tolist() # << list로 변환한 것
df3.to_dict('list')


# ### DataFreame의 칼럼 데이터 세트 생성과 수정

# In[9]:


titanic_df['Age_0']=0
titanic_df.head(3)


# In[10]:


titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No']=titanic_df['SibSp']+titanic_df['Parch']+1


# - dataframe 데이터 삭제( drop() ) >> 중요 param : labels, axis, inplace
# 
# lables : 삭제하고자 하는 컬럼
# axis : 보통 칼럼을 drop하기 때문에 axis = 1을 많이 쓰지만 이상치를 제거하고자 하면 axis = 0으로 제거하기
# inplace = True : drop한 결과 -> 원본에 반영 / default 값은 inplace = False

# In[11]:


titanic_drop_df = titanic_df.drop('Age_0',axis =1)
titanic_drop_df.head(3)


# - 데이터 삭제 (______.drop([index번호], axis = 0)
# -인덱스만 추출 : ______.index.values
# -titanic_reset_df = titanic_df.reset_index(inplace = False) >> index라는 컬럼이 따로 생김 -> type is a dataframe.

# - 데이터 selection and filtering

# In[15]:


#iloc[]
data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
data_df = pd.DataFrame(data, index=['one','two','three','four'])
data_df


# In[16]:


data_df.iloc[0,1] # int형으로만 입력(위치 기반 데이터 추출)


# In[21]:


#loc (위치 기반형이라서 슬라이싱할 때 마지막 포함)
print(data_df.loc['one','Name'])


# ##### 불린 인덱싱

# In[24]:


titanic_boolean = titanic_df[titanic_df['Age']>60]
titanic_boolean.head(3)


# In[25]:


titanic_df[titanic_df['Age'] > 60][['Name','Age']].head(3)


# In[26]:


titanic_df.loc[titanic_df['Age'] > 60, ['Name','Age']].head(3) # 위와 같은 결과


# ### 정렬, Aggregation 함수, GroupBy 적용
# 
# 1. 정렬(sort_values())
# - by = ['칼럼명'] : 칼럼으로 정렬 수행
# - ascending : default는 True(오름차순)
# - inplace : default는 False

# In[27]:


titanic_sorted = titanic_df.sort_values(by =['Name'])
titanic_sorted.head(3)


# In[29]:


titanic2 = titanic_df.sort_values(by = ['Pclass','Name'], ascending = False)
titanic2.head(3)


# 2. Aggregation

# In[30]:


titanic_df.count()


# In[32]:


titanic_df[['Age','Fare']].mean()


# 3. GroupBy

# In[34]:


titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId','Survived']].count()
titanic_groupby.head()


# In[35]:


titanic_df.groupby('Pclass')['Age'].agg([max,min])  # 이런거 보면 SQL이 유연성 면에서는 탁월 (열마다 다른 aggregation을 취하니까)


# In[36]:


# 위의 문제를 해결하려면
agg_format = {'Age' : 'max', 'SibSp' : 'sum', 'Fare' : 'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# ### 결손 데이터 처리하기
# 
# 
# NaN이 존재하면 : mean 값은 결손값 제외한 갯수로 구함
# 
# 1. 결손값 확인 : isna()

# In[37]:


titanic_df.isna().head(3)


# In[39]:


titanic_df.isna().sum()


# 2. 결손 데이터 대체 : fillna()

# In[42]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000') # <------ 원본 데이터에 반영하려면 inplace = True 쓰기
titanic_df.head(3)


# In[45]:


titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# ### apply lambda 식으로 데이터 가공

# In[46]:


lambda_square=lambda x : x**2
print(lambda_square(3))


# In[49]:


# 여러개 인자를 입력 받으면?
a= [1,2,3]
squares = map(lambda x : x**2,a)
list(squares)


# In[50]:


# 이름의 길이 구하기
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name','Name_len']].head(3)


# In[51]:


# 조금 더 복잡하게
titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult')
titanic_df[['Age','Child_Adult']].head(3)

