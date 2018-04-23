import numpy as np
#importing numpy

#defination of Kernel function
def Kernel (X,Y):
    return np.square(1+np.transpose(X).dot(Y))
#defination of Phai(a non linear transformation) function 
def Phai (x):
    return np.array([1,np.square(x[0]),np.sqrt(2)*x[0]*x[1],np.square(x[1]),np.sqrt(2)*x[0],np.sqrt(2)*x[1]])
#Training Set: Each row contains feature vector and class label.Last column contains class label.
T=np.array([[-1,-1,-1], [-1,1,1],[1,-1,1],[1,1,-1]])
print( 'Training set :\n')
print('---------------\n')
print T
N = T.shape[0]
n = T.shape[1]  
M=np.zeros((N,N))         
for i in range(N):
    for j in range(N):
        M[i][j]=T[i][n-1]*T[j][n-1]*Kernel (np.transpose(T[i][0:-1]),np.transpose(T[j][0:-1]))
   
#M_inv=inverse = np.linalg.inv(M)   
M_inv=np.linalg.inv(M)

Lamda=M_inv.dot(np.ones((N,1)))
print('\n Values of Lamda : \n')
print Lamda
print('\n Values of W : \n')
#pdb.set_trace()
W=np.zeros((1,6))
print W
print('\n Values after transpose : \n')
for i in range(N):
    W=W+np.array(T[i][n-1]*Lamda[i]*Phai(T[i][0:-1]))
    #pdb.set_trace()
    
print np.transpose(W)
b=np.zeros((4,1))
Sum_b=0
for i in range(N):
    b[i]=T[i][n-1]-W.dot(Phai(T[i][0:-1]))
    Sum_b=Sum_b+b[i]

mean_b=Sum_b/N
print'\n Values of b : \n'
print b
print('\n Value of b_mean : \n')
print mean_b

Tst=np.array([[0.9,-0.8,1],[-0.5,-0.7,-1]])
N = Tst.shape[0]
n = Tst.shape[1]
count=0
for i in range(N):
    predict=np.sign(W.dot(Phai(Tst[i][0:-1]))+mean_b)
    if predict==Tst[i][n-1]:
        count=count+1

accu=(count*100)/N
print 'Percentage of accuracy -->',accu
#print accu

