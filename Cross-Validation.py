#NAME: STEFANIA MAKRYGIANNAKI
#AM: 164703

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Διάβασμα του αρχείου δεδομένων iris.data στην Python
data = read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values

# Πλήθος δειγμάτων       = NumberOfPatterns   = 150
# Πλήθος χαρακτηριστικών = NumberOfAttributes = 5
NumberOfPatterns, NumberOfAttributes = data.shape
print(NumberOfAttributes, NumberOfPatterns)

# Αντιγραφή των 4ων στηλών του data στο x
x = data[0:NumberOfPatterns,0:4]

fig = plt.figure()

# Η θέση του τελευταίου στοιχείου του Setosa
numberOfSetosa = np.argmax(np.where(data=='Iris-setosa'))
# Η θέση του τελευταίου στοιχείου του Versicolor
numberOfVersicolor = np.argmax(np.where(data=='Iris-versicolor')) + numberOfSetosa + 1
# Η θέση του τελευταίου στοιχείου του Virginica
numberOfVirginica = np.argmax(np.where(data=='Iris-virginica')) + numberOfVersicolor + 1

# Ο πρώτος πίνακας
plt.plot(x[0:numberOfSetosa,0],x[0:numberOfSetosa,2],'r*')
plt.plot(x[numberOfSetosa+1:numberOfVersicolor,0],x[numberOfSetosa+1:numberOfVersicolor,2],'g+')
plt.plot(x[numberOfVersicolor+1:numberOfVirginica,0],x[numberOfVersicolor+1:numberOfVirginica,2],'yd')
plt.title('1st')
plt.show()

# Αρχικοποίηση του πίνακα t
t = np.zeros((NumberOfPatterns,1))

# Menu
ans = "y"
while ans == "y":
    print("1 Διαχωριςμόσ Iris-setosa από Iris-versicolor και Iris-virginica")
    print("2 Διαχωριςμόσ Iris-virginica από Iris-setosa και Iris-versicolor")
    print("3 Διαχωριςμόσ Iris-versicolor από Iris-virginica και Iris-setosa")
    answer = raw_input('Select: ')

    # Επιλογές
    if answer == '1':
        map_dict={"Iris-setosa": 1, "Iris-versicolor": 0, "Iris-virginica": 0}
    elif answer == '2':
        map_dict={"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 1}
    elif answer == '3':
        map_dict={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0}
    else:
        print('Διάλεξε μία απο τις επιλογές 1 εως 3 \n')
        continue

    # Θέτω την τιμή στόχου t με βάση την επιλογή του χρήστη
    for i in range(NumberOfPatterns):
        t[i] = map_dict[data[i,4]]
    
    # Δημιουργία πινάκων xtrain, xtest, ttrain, ttest
    xtrain = np.vstack((x[0:40],x[50:90],x[100:140]))   
    xtest = np.vstack((x[40:50],x[90:100],x[140:150]))    
    ttrain = np.vstack((t[0:40],t[50:90],t[100:140]))   
    ttest = np.vstack((t[40:50],t[90:100],t[140:150]))
    
    # Ο δεύτερος πίνακας
    fig2 = plt.figure()   
    plt.plot(xtrain[:,0],xtrain[:,2],'b.')
    plt.plot(xtest[:,0],xtest[:,2],'r.')
    plt.title('2nd')   
    plt.show()
    
    # Ο τρίτος πίνακας, ομάδα 9 πινάκων
    fig3 = plt.figure()
    fig3.subplots_adjust(hspace=0.5, wspace=0.4)
    
    for folds in range(9):
        xtrain, xtest, ttrain, ttest = train_test_split(
                x, t, test_size = 0.1)
        ax = fig3.add_subplot(3,3,folds+1)
        ax.plot(xtrain[:,0],xtrain[:,2],'b.')
        ax.plot(xtest[:,0],xtest[:,2],'r.')
        ax.set_title('Fold %i' %folds)
    
    plt.show() 
    
    ans = raw_input('Do you want to continue? [type "y" for "yes", anything else for no]: ')
