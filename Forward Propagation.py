#!/usr/bin/env python
# coding: utf-8

# # Forward Propagation
# ### Propagating activations forward through the network. Activations are all the numbers that are not weights and are unique for every prediction.

# ### Neural network using single parameter

# In[2]:


weight = 0.1
def neural_network(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5, 9.5, 10, 9] #Average number of toes per player
for x in number_of_toes:
    pred = neural_network(x, weight)
    print(pred)


# ### Basic neural network using multiple parameters

# In[3]:


weights = [0.1, 0.2, 0]

def w_sum(a, b):
    assert(len(a) == len(b)) #Checks if the statement is true, if not throws an Assertion error
    output = 0
    for i in range(len(a)):
        output = output + (a[i] * b[i]) #Weighted sum or Elementwise dot product
    return output
    
def neural_network(input, weights):
    pred = w_sum(input, weights)
    print(pred)

toes = [8.5, 9.5, 9.9, 9]#Average number of toes per player
wlrec = [0.65, 0.8, 0.8, 0.9]#Current games won(percent)
nfans = [1.2, 1.3, 0.5, 1.0]#Fan count(in millions)

for x in range(0, 4):
    input = [toes[x], wlrec[x], nfans[x]]
    neural_network(input, weights)    


# ### Using NumPy, making a basic neural network with multiple parameters

# In[4]:


import numpy as np
weights = np.array([0.1, 0.2, 0])
def neural_network(input, weights):
    pred = np.dot(input, weights)
    print(pred)

toes = np.array([8.5, 9.5, 9.9, 9]) #Average number of toes per player
wlrec = np.array([0.65, 0.8, 0.8, 0.9]) #Current games won(percent)
nfans = np.array([1.2, 1.3, 0.5, 1.0]) #Fan count(in millions)

for x in range(0, 4):
    input = [toes[x], wlrec[x], nfans[x]]
    neural_network(input, weights)


# ### Predicting with multiple inputs and outputs

# In[5]:


weights = [[0.1, 0.1, -0.3], 
           [0.1, 0.2, 0.0], 
          [0.0, 1.3, 0.1]] #hurt, win, sad

def w_sum(a, b):
    assert(len(a) == len(b)) #Checks if the statement is true, if not throws an Assertion error
    output = 0
    for i in range(len(a)):
        output = output + (a[i] * b[i]) #Weighted sum or Elementwise dot product
        print(a[i], '*', b[i], '=', output)
    return output

def vect_mat_mul(vect, matrix): #this function iterates through each row of weights and makes a prediction using w_sum fn
    assert(len(vect) == len(matrix))
    output = [0, 0, 0]
    
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])
        if i == 0:
            print("Prediction of being hurt = ", output[i])
        elif i == 1: 
            print("Prediction of winning = ", output[i])
        else: 
             print("Prediction of being sad = ", output[i])
    return output

def neural_network(input, weights):
    predict = vect_mat_mul(input, weights)
    return pred
    
toes = [8.5, 9.5, 9.9, 9]#Average number of toes per player
wlrec = [0.65, 0.8, 0.8, 0.9]#Current games won(percent)
nfans = [1.2, 1.3, 0.5, 1.0]#Fan count(in millions)

for x in range(0, 4):
    input = [toes[x], wlrec[x], nfans[x]]
    print("Game", (x + 1), "of season")
    neural_network(input, weights)
    print('\n')


# ### Predicting on predictions

# In[12]:


#toes  #wins  #fans
ih_wgt = [[0.1, 0.2, -0.1],
 [-0.1, 0.1, 0.9],
 [0.1, 0.4, 0.1]]
#hid[0]  #hid[1]  #hid[2]
hp_wgt = [[0.3, 1.1, -0.3],
  [0.1, 0.2, 0.0],
  [0.0, 1.3, 0.1]]

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    print("Hidden network")
    hid = vect_mat_mul(input, weights[0])
    print("Final layer/ Output")
    pred = vect_mat_mul(hid, weights[1])
    print(pred)
    
toes = [8.5, 9.5, 9.9, 9]#Average number of toes per player
wlrec = [0.65, 0.8, 0.8, 0.9]#Current games won(percent)
nfans = [1.2, 1.3, 0.5, 1.0]#Fan count(in millions)

for x in range(0, 4):
    input = [toes[x], wlrec[x], nfans[x]]
    print("Game", (x + 1), "of season")
    neural_network(input, weights)
    print('\n')


# ### Predicting on predictions using numPy

# In[11]:


import numpy as np
#toes #wins #fans
ih_wgt = np.array([[0.1, 0.2, -0.1],
         [-0.1, 0.1, 0.9],
         [0.1, 0.4, 0.1]]).T
        #hid[0]  #hid[1]  #hid[2]
hp_wgt = np.array([[0.3, 1.1, -0.3],
          [0.1, 0.2, 0.0],
          [0.0, 1.3, 0.1]]).T

def neural_network(input, weights):
    print("Hidden network")
    hid = np.dot(input, weights[0])
    print(hid)
    print("Final layer/output")
    pred = np.dot(hid, weights[1])
    print(pred)
    
weights = [ih_wgt, hp_wgt]

toes = [8.5, 9.5, 9.9, 9]#Average number of toes per player
wlrec = [0.65, 0.8, 0.8, 0.9]#Current games won(percent)
nfans = [1.2, 1.3, 0.5, 1.0]#Fan count(in millions)

for x in range(0, 4):
    input = [toes[x], wlrec[x], nfans[x]]
    print("Game", (x + 1), "of season")
    neural_network(input, weights)
    print('\n')


# ## Numpy quick tutorial on vectors and matrices

# ### To perform matrix multiplication, the column number of the first matrix should match the row number of the second matrix

# In[26]:


import numpy as np
a = np.array([0, 1, 2, 3])#vector
b = np.array([4, 5, 6, 7])#vector
c = np.array([[0, 1, 2, 3],
            [4, 5, 6, 7]])#matrix
d = np.zeros((2, 4))
e = np.random.rand(2, 5)
print('Vector a ', a)
print('Vector b ', b)
print('Matrix c ', c)
print('Matrix d ', d)
print('Matrix e ', e)
print(a * 0.1)
print(c * 0.2)
print(a * b)
print(a * b * 0.2)
a = np.ones((2, 4))
b = np.ones((4, 3))
c = np.dot(a, b)
c.shape


# In[ ]:




