import pandas as pd
import numpy as np
foxconndf= pd.read_csv('FINAL_DATA.csv')
foxconndf.dropna(how='any',inplace=True)

import numpy as np
from sklearn.model_selection import train_test_split
# seed = 90 94.77%
seed = 90
test_size = 0.3
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def splitXy(foxconndf):
    dataset = foxconndf.values
    
    X = dataset[:,0:-1]
    y = dataset[:,-1]
    return X, y

# split data into X and y
X, y = splitXy(foxconndf)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=seed)
# 95.68留存
model2 =SVR(kernel ='rbf',gamma='scale', C=3, epsilon=0.2)
model3 = KNeighborsRegressor(n_neighbors=20,algorithm='auto')
history2 = model2.fit(X_train, y_train)
history3 = model3.fit(X_train, y_train)

import pickle
pickle.dump(history2, open("SVR_model.pkl", "wb"))
pickle.dump(history3, open("KNN_model.pkl", "wb"))


ans2 = model2.predict(X_test)
ans3 = model3.predict(X_test)

    
    # 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    nnn=round(((ans2[i]+ans3[i])/2),0)
#         print("預測值",nnn,"實際值",round(y_test[i],0))
    if nnn == round(y_test[i],0):
        cnt1 += 1
    elif (nnn+1) == round(y_test[i],0):
        cnt1 += 1
    elif (nnn-1) == round(y_test[i],0):
        cnt1 += 1
    elif (nnn+2) == round(y_test[i],0):
        cnt1 += 1
    elif (nnn-2) == round(y_test[i],0):
        cnt1 += 1
    else:
        cnt2 += 1
print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

