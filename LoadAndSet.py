import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

data=np.load('/CASIA 2.0/data.npy')
target=np.load('/CASIA 2.0/target.npy')

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.3,stratify=target,random_state=5)

np.random.seed(5)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=5)

train_lt=[]
test_lt=[]
for train,test in kfold.split(data,target.argmax(1)):
  train_lt.append(train)
  test_lt.append(test)
len(train_lt)
len(train_lt),len(test_lt),len(train_lt[0]),len(test_lt[0])
