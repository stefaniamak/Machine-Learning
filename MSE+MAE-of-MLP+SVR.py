#NAME: STEFANIA MAKRYGIANNAKI
#AM: 164703
#ΤΜΗΜΑ: T3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


''' -------------------- Συναρτήσεις -------------------- '''

def regrevaluate(tt, predictt, criterion):
# Είσοδοι: 
#   t : διάνυσμα με τους πραγματικούς στόχους (πραγματικοί αριθμοί) 
#   predict : διάνυσμα με τους εκτιμώμενους στόχους (πραγματικοί αριθμοί) 
#   criterion : text-string με τις εξής πιθανές τιμές:
#       'mse' 
#       'mae' 
# Έξοδος value : η τιμή του κριτηρίου που επιλέξαμε.    
    
    if criterion == 'mse':
        value = calculation(tt,predictt,2)
    else:
        value = calculation(tt,predictt,1)
    return value

def calculation(tt,predictt,dynami):
    n = len(predictt)
    value = 0.0
    for i in range(n):
        value += abs(pow((tt[i] - predictt[i]),dynami))    
    value = (value / (n*1.0)) + 0.0
    return value



''' -------------- ΤΕΛΟΣ ΣΥΝΑΡΤΗΣΕΩΝ ------------- '''



''' -------------------- MAIN -------------------- '''

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, delim_whitespace=True).values
NumberOfPatterns, NumberOfAttributes = data.shape

#Αρχικοποιώ τον πίνακα χ
x = np.zeros((NumberOfPatterns,(NumberOfAttributes-1))) #htan anapoda
t = np.zeros((NumberOfPatterns))

# Αντιγραφή των 12ων στηλών του data στο x κ' 13ης στο t
x = data[:,0:13].astype(float)
t = data[:,13].astype(float)

 

''' -------------------- SVR -------------------- '''


print('\n---------------------- SVR ----------------------')
# Αρχικοποιήσεις των min
mse = np.zeros(9)
minMSE = 99999999.0
minCmse = 0.0
minGmse = 0.0
    
mae = np.zeros(9)
minMAE = 99999999.0
minCmae = 0.0
minGmae = 0.0
    
fig12 = plt.figure()
fig12.subplots_adjust(hspace=0.5, wspace=0.4)
            
for gLoop in [0.0001,0.001,0.01,0.1]:
    for cLoop in [1,10,100,1000]:         
        for folds in range(9):      
            
            xtrain, xtest, ttrain, ttest = train_test_split(
                    x, t, test_size = 0.1)    
            
            nety = SVR(C=cLoop,
                          kernel='rbf',
                          gamma=gLoop)
           
            nety.fit(xtrain,ttrain)
            predict = nety.predict(xtest)
              
            mse[folds] = regrevaluate(ttest, predict, 'mse')
            mae[folds] = regrevaluate(ttest, predict, 'mae')

        if (mse.mean() < minMSE):
            minMSE = mse.mean()
            minCmse = cLoop
            minGmse = gLoop
                    
        if (mae.mean() < minMAE):
            minMAE = mae.mean()
            minCmae = cLoop
            minGmae = gLoop


''' ---------- MSE ---------- '''


print 'Smallest Mean Square Error  :',minMSE, '| C=',minCmse, 'G=', minGmse
xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size = 0.1) 
SVM = SVR(C=minCmse,
          kernel='rbf',
          gamma=minGmse)
                      
# Εκπαίδευση του δικτύου
SVM.fit(xtrain, ttrain)
# Ανάκληση
predictTest = SVM.predict(xtest) 
    
ax = fig12.add_subplot(1,2,1)
ax.plot(ttest[:], 'b-')
ax.plot(predictTest[:], 'r+') 
ax.set_title('mse')
 


''' ---------- MAE ---------- '''


print 'Smallest Mean Absolute Error:',minMAE, '| C=',minCmae, 'G=', minGmae
xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size = 0.1) 
SVM = SVR(
        C=minCmae,
        kernel='rbf',
        gamma=minGmae)
                      
# Εκπαίδευση του δικτύου
SVM.fit(xtrain, ttrain)
# Ανάκληση
predictTest = SVM.predict(xtest) 
    
ax = fig12.add_subplot(1,2,2)
ax.plot(ttest[:], 'b-')
ax.plot(predictTest[:], 'r+')
ax.set_title('mae')
 
   
plt.show() 




''' -------------------- MLP -------------------- '''


print('\n---------------------- MLP ----------------------')

fig34 = plt.figure()
fig34.subplots_adjust(hspace=0.5, wspace=0.4)

mse = np.zeros(9)
minMSE = 99999999.0
minNmse = 0.0
    
mae = np.zeros(9)
minMAE = 99999999.0
minNmae = 0.0


for N in [5,10,20,30,40,50]:
    '''FOLDS'''    
    for folds in range(9):      
        xtrain, xtest, ttrain, ttest = train_test_split(
                x, t, test_size = 0.1) 

        # Πλήθος προτύπων 
        ptrain = len(ttrain)
        ptest = len(ttest)

        # Δημιουργία δικτύου MLP δύο στρωμάτων
        MLP = MLPRegressor(hidden_layer_sizes=(N), 
                           activation='relu', 
                           solver='adam', 
                           max_iter=200, 
                           learning_rate='constant', 
                           learning_rate_init=0.001,
                           momentum=0.9)


        # Εκπαίδευση του MLP δικτύου
        MLP.fit(xtrain, ttrain)
        # Ανάκληση
        predictTest = MLP.predict(xtest)    
        
        mse[folds] = regrevaluate(ttest, predict, 'mse')
        mae[folds] = regrevaluate(ttest, predict, 'mae')
               
             
    if (mse.mean() < minMSE):
        minMSE = mse.mean()
        minNmse = N;
                    
    if (mae.mean() < minMAE):
        minMAE = mae.mean()
        minNmae = N;

''' ---------- MSE ---------- '''  


print 'Smallest Mean Square Error  :',minMSE, '| N=',minNmse
xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size = 0.1) 

MLP = MLPRegressor(hidden_layer_sizes=(N), 
                   activation='relu', 
                   solver='adam', 
                   max_iter=200, 
                   learning_rate='constant', 
                   learning_rate_init=0.001,
                   momentum=0.9)
                      
# Εκπαίδευση του MLP δικτύου
MLP.fit(xtrain, ttrain)
# Ανάκληση
predictTest = MLP.predict(xtest) 
    
ax = fig34.add_subplot(1,2,1)
ax.plot(ttest[:], 'b-')
ax.plot(predictTest[:], 'r+')
ax.set_title('mse')
 


''' ---------- MAE ---------- '''


print 'Smallest Mean Absolute Error:',minMAE, '| N=',minNmae
xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size = 0.1) 

MLP = MLPRegressor(hidden_layer_sizes=(N), 
                   activation='relu', 
                   solver='adam', 
                   max_iter=200, 
                   learning_rate='constant', 
                   learning_rate_init=0.001,
                   momentum=0.9)
                      
# Εκπαίδευση του MLP δικτύου
MLP.fit(xtrain, ttrain)
# Ανάκληση
predictTest = MLP.predict(xtest) 
    
ax = fig34.add_subplot(1,2,2)
ax.plot(ttest[:], 'b-')
ax.plot(predictTest[:], 'r+')
ax.set_title('mae')
 

plt.show()     
    

print('\n-------------------- THE END ---------------------' + '\033[0m')

''' -------------------- END MAIN -------------------- '''





