import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('./knnMNIST/train.csv')
data=df.values
# print(data.shape)

X=data[:,1:]
Y=data[:,0]
# print(X.shape)
# print(Y.shape)


split=int(0.8*X.shape[0])
x_train=X[:split,:]
y_train=Y[:split]

x_test=X[split:,:]
y_test=Y[split:]

# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x_train,y_train,x_test,k=5):
    dist_val=[]
    m=x_train.shape[0]
    for ix in range(m):
        d=distance(x_test,x_train[ix])
        dist_val.append([d,y_train[ix]])

    dist_val=sorted(dist_val)
    dist_val=dist_val[:k]

    y=np.array(dist_val)
    ans=np.unique(y[:,1],return_counts=True)
    index=ans[1].argmax()
    prediction=ans[0][index]
    return int(prediction)

def draw_image(x):
    img=x.reshape((28,28))
    plt.imshow(img)

query_x=x_test[4]
print(y_test[4])
print(knn(x_train,y_train,query_x))
draw_image(query_x)
plt.show()

count=0
for i in range(100):
    if knn(x_train,y_train,x_test[i]) == y_test[i]:
        count+=1
print(count/100)


