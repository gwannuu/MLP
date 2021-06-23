import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

trains=np.loadtxt("Trn.txt")
tests=np.loadtxt("Tst.txt")

np.set_printoptions(precision=4)
df_train=pd.read_csv("Trn.txt",sep=" ",header=None)

class HiddenLayer:
    def __init__(self,inputsize,neuronNum,r):
        self.neuronNum=neuronNum
        self.b=np.random.randn(neuronNum)
        self.r=r # learning rate
        self.W=np.random.uniform(-1*np.sqrt(6/(inputsize+neuronNum)),np.sqrt(6/(inputsize+neuronNum)),size=(neuronNum,inputsize)) #xavier initializer

    def feedforward(self,x):
        self.x=x
        self.z=np.matmul(self.W,self.x)+self.b
        self.a=1/(1+np.exp(-1*self.z))
        return self.a
    
    def backpropagation(self,dlda):
        dadz=self.a*(1-self.a)
        dldz=np.multiply(dlda,dadz)
        dldx=np.matmul(dldz,self.W) #dldz*dzdx              
        dldW=np.outer(dldz,self.x)                       
        dldb=dldz*np.ones(self.b.shape[0])
        self.W=self.W-self.r*dldW
        self.b=self.b-self.r*dldb
        return dldx
            
class OutputLayer:
    def __init__(self,inputSize,learningRate):
        self.W=np.random.uniform(-1*np.sqrt(6/(inputSize+1)),np.sqrt(6/(inputSize+1)),size=inputSize) #xavier initializer
        self.b=np.random.randn(1)
        self.r=learningRate
        
    def feedforward(self,x):
        self.x=x
        self.z=np.dot(self.W,self.x)+self.b
        self.a=1/(1+np.exp(-1*self.z))
        return self.a
    
    def loss(self,y):
        self.y=y
        loss=(self.a-self.y)**2
        return loss

    def backpropagation(self):
        dlda=1*(self.a-self.y)
        dadz=self.a*(1-self.a)
        dzdx=self.W
        dzdW=self.x
        dzdb=1
        dldW=dlda*dadz*dzdW
        dldb=dlda*dadz*dzdb
        self.W=self.W-self.r*dlda*dadz*dzdW
        self.b=self.b-self.r*dlda*dadz*dzdb
        return dlda*dadz*dzdx
        
      
class MLP:
    def __init__(self):
        self.H1=HiddenLayer(2,12,0.6)
        self.H2=HiddenLayer(12,4,0.6)
        self.O=OutputLayer(4,0.6)
    
    def feedforward(self,x,y):
        a1=self.H1.feedforward(x)
        a2=self.H2.feedforward(a1)
        self.O.feedforward(a2)
        self.O.loss(y)
    
    def backpropagation(self):
        
        dlda1=self.O.backpropagation()
        dlda=self.H2.backpropagation(dlda1)
        self.H1.backpropagation(dlda)
        
    def classification(self,x):
        a1=self.H1.feedforward(x)
        a2=self.H2.feedforward(a1)
        return self.O.feedforward(a2)[0]

mymlp=MLP()
count=1

for _ in range(1000000):
    if count%10000==0:
        print(count/10000,"% training done")
    i=np.random.randint(0,630)
    mymlp.feedforward(trains[i,0:2],trains[i,2])
    mymlp.backpropagation()
    count+=1
    
index=[]
for i in tests:
    pred=mymlp.classification(i)
    index.append([i[0],i[1],pred])

df=pd.DataFrame(index)

sns.scatterplot(x=0,y=1,hue=2,data=df,legend=False)
plt.scatter(df_train[0][df_train[2]==0],df_train[1][df_train[2]==0])
plt.scatter(df_train[0][df_train[2]==1],df_train[1][df_train[2]==1])
plt.savefig("./Testing visualization with test data.png")

sns.scatterplot(x=0,y=1,hue=2,data=df,legend=False)
plt.savefig("./Testing visualization.png")
