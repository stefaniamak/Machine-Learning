#NAME: STEFANIA MAKRYGIANNAKI
#AM: 164703
#ΤΜΗΜΑ: T3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


''' -------------------- MAIN -------------------- '''

# Διάβασμα του αρχείου δεδομένων iris.data στην Python
data = np.load('mnist_49.npz')

x = data['x'] # Εικόνα διάστασης 28×28 με 784 pixels (784=28×28)
t = data['t'] # Αν ο στόχος είναι 0 τότε το πρότυπο είναι το ψηφίο “4” αλλιώς είναι το ψηφίο “9”

 
print('\033[31m'+'\n Ανάλυσης Κυρίων Συνιστωσών (Principal Component Analysis - PCA)' + '\033[0m')
#Σκοπός της άσκησης: η χρήση της Ανάλυσης Κυρίων Συνιστωσών (Principal Component 
#Analysis - PCA) για την συμπίεση των δεδομένων εισόδου και η μελέτη της 
#επίπτωσης αυτής της ανάλυσης στην επίδοση ενός μοντέλου Μηχανικής Μάθησης. 
#Ως παράδειγμα θα χρησιμοποιηθεί ένα μοντέλο Naïve Bayes με Γκαουσσιανή 
#συνάρτηση πυκνότητας πιθανότητας. 
print

acc_train_folds = np.arange(10).astype(float)
acc_test_folds = np.arange(10).astype(float)


#''' ---------- WITHOUT PCA ---------- '''

print '\033[34m' 
print '       {}'.format('Results withough PCA:')
print '\033[0m'

'''FOLDS'''  
for folds in range(10):      
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size = 0.1) 
    
    # Ταξινόμηση των προτύπων με την μέθοδο Naïve Bayes/Gaussian
    model = GaussianNB() # Δημιουργ'ια μοντέλου
    model.fit(xtrain, ttrain) # Εκπαίδευση
    
    # Αξιολόγηση / Υπολογισμός accuracy
    acc_train_folds[folds] = model.score(xtrain, ttrain)
    acc_test_folds[folds] = model.score(xtest, ttest)
      
print 'Mean Train Accuracy:', np.mean(acc_train_folds)
print 'Mean Test Accuracy:', np.mean(acc_test_folds)


#''' ---------- WITH PCA ---------- '''

print
print '\033[34m' 
print '         {}'.format('Results with PCA:')
print '\033[0m'

acc_train = np.arange(10).astype(float)
acc_test = np.arange(10).astype(float)

# Πλήθος χαρακτηριστικών (number of components)
num_components = np.array([1, 2, 5, 10, 20, 30, 40, 50, 100, 200])

for i in  range(len(num_components)):
    pca = PCA(n_components = num_components[i]) # Δημιουργία μοντέλου PCA
    x_pca = pca.fit_transform(x) # Πίνακας συμπιεσμένων δεδομένων
    '''FOLDS'''  
    for folds in range(10):      
        xtrain, xtest, ttrain, ttest = train_test_split(x_pca, t, test_size = 0.1) 
    
        modelPCA = GaussianNB() # ταξινόμηση με το μοντέλο Naïve Bayes/Gaussian
        modelPCA.fit(xtrain, ttrain) 
    
        acc_train_folds[folds] = modelPCA.score(xtrain, ttrain)
        acc_test_folds[folds] = modelPCA.score(xtest, ttest)
        
    acc_train[i] = np.mean(acc_train_folds)
    acc_test[i] = np.mean(acc_test_folds)
     
fig = plt.figure() # Γράφημα
plt.plot(num_components[:],acc_train[:],'r-', label='train')
plt.plot(num_components[:],acc_test[:],'b-', label='test')
plt.title('Naive Bayes')
plt.xlabel('Number of PCA components')
plt.ylabel('Accuracy')
legend = plt.legend(loc='upper right', shadow=False, fontsize='x-large')
plt.show()
        
        

print('\033[31m'+'\n-------------------- THE END ---------------------' + '\033[0m')
''' -------------------- END MAIN -------------------- '''





