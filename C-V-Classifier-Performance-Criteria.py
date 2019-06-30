#NAME: STEFANIA MAKRYGIANNAKI
#AM: 164703
#ΤΜΗΜΑ: T3

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


''' -------------------- Συναρτήσεις -------------------- '''

''' Έλεγχος των στοιχείων από τη συνάρτηση evaluate πριν την καταμέτρηση τους '''
def countEval(n,q,ttest,predict,criterion):
    temp = evaluate(ttest, predict, criterion)
    if (temp <= 1.0 and temp >=0):
        n = n + temp
        q = q + 1.0
    return n,q


''' Evaluate '''
def evaluate( t, predict, criterion):
    """ 
    Είσοδος t : διάνυσμα με τους πραγματικούς στόχους (0/1) 
    Είσοδος predict : διάνυσμα με τους εκτιμώμενους στόχους (0/1) 
    Είσοδος criterion : text-string με τις εξής πιθανές τιμές:   
        'accuracy','precision','recall','fmeasure','sensitivity','specificity'
    Έξοδος value : η τιμή του κριτηρίου που επιλέξαμε. 
    """ 
    
    # true positives
    tp = truefalse(predict,t,1,1)
    # false positives
    fp = truefalse(predict,t,1,0)
    # true negatives
    tn = truefalse(predict,t,0,0)
    # false negatives
    fn = truefalse(predict,t,0,1)
    
    if criterion == 'accuracy':
        value = accuracy(tp,fp,tn,fn)
    elif criterion == 'precision':
        value = precision(tp,fp)
    elif criterion == 'recall':
        value = recall(tp,fn)
    elif criterion == 'fmeasure':
        value = fmeasure(tp,fp,tn,fn)       
    elif criterion == 'sensitivity':
        value = sensitivity(tp,fn)
    elif criterion == 'specificity':
        value = specificity(fp,tn)   
    return value


''' Συνάρτηση για τις τιμές tp,fp,tn,fn '''
def truefalse(predict, t, n1, n2):
    first = (predict==n1)
    second = (t==n2)
    third = np.logical_and(first, second)
    return((sum(third)).astype(float))   

''' Συναρτήσεις για τις τιμές του criterion '''
def accuracy(tp,fp,tn,fn):
    if((tp+tn+fp+fn) == 0):
        return(-50.0)
    else:
        return((tp + tn) / (tp + tn + fp + fn))
        
def precision(tp,fp):
    if((tp + fp) == 0):
        return(-50.0)
    else:
        return((tp / (tp + fp)))
        
def recall(tp,fn):
    if((tp + fn) == 0):
        return(-50.0)
    else:
        return((tp / (tp + fn)))
        
def fmeasure(tp,fp,tn,fn):
    pres = precision(tp,fp)
    rec = recall(tp,fn)
    if(np.logical_or(pres+rec==0,pres+rec== -100)):
        return(-50.0)
    else:
        return(((pres * rec) / ((pres + rec)/2)))
        
def sensitivity(tp,fn):
    return((recall(tp,fn)))
        
def specificity(fp,tn):
    if((tn + fp) == 0):
        return(-50.0)
    else:
        return((tn / (tn + fp)))     


''' -------------- ΤΕΛΟΣ ΣΥΝΑΡΤΗΣΕΩΝ ------------- '''



''' -------------------- MAIN -------------------- '''

# Διάβασμα του αρχείου δεδομένων iris.data στην Python
data = read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None).values

# Πλήθος δειγμάτων       = NumberOfPatterns   = 150
# Πλήθος χαρακτηριστικών = NumberOfAttributes = 5
NumberOfPatterns, NumberOfAttributes = data.shape

# Αντιγραφή των 4ων στηλών του data στο x
x = data[0:NumberOfPatterns,0:4].astype(float)

fig = plt.figure()

# Ο πρώτος πίνακας
plt.plot(x[0:50,0],x[0:50,2],'r*')
plt.plot(x[50:100,0],x[50:100,2],'g+')
plt.plot(x[100:150,0],x[100:150,2],'yd')
plt.title('1st')
plt.show()

# Αρχικοποίηση του πίνακα t
t = np.zeros((NumberOfPatterns,1))

# Menu
ans = "y"
while ans == "y":
    print('\n------------------------------MENU------------------------------')
    print('\033[35m')
    print('1. Διαχωρισμός Iris-setosa από Iris-versicolor και Iris-virginica')
    print('2. Διαχωρισμός Iris-virginica από Iris-setosa και Iris-versicolor')
    print('3. Διαχωρισμός Iris-versicolor από Iris-virginica και Iris-setosa')
    print('\033[0m')
    answer = raw_input('Select: ')

    # Επιλογές
    if answer == '1':
        map_dict={"Iris-setosa": 1, "Iris-versicolor": 0, "Iris-virginica": 0}
    elif answer == '2':
        map_dict={"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 1}
    elif answer == '3':
        map_dict={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0}
    else:
        print('\033[31m' +'Διάλεξε μία απο τις επιλογές 1 εως 3 \n'+ '\033[0m')
        continue

    # Θέτω την τιμή στόχου t με βάση την επιλογή του χρήστη
    for i in range(NumberOfPatterns):
        t[i] = map_dict[data[i,4]]
    
    # Επαυξάνω τον πίνακα των προτύπων προσθέτοντας μια γραμμή με 1 
    ones = np.ones((150,1))
    x = np.hstack((x,ones)) 
    
    # Δημιουργία πινάκων xtrain, xtest, ttrain, ttest
    xtrain = np.vstack((x[0:40],x[50:90],x[100:140])).astype(float)  
    xtest = np.vstack((x[40:50],x[90:100],x[140:150])).astype(float)    
    ttrain = np.vstack((t[0:40],t[50:90],t[100:140])).astype(float)  
    ttest = np.vstack((t[40:50],t[90:100],t[140:150])).astype(float) 
    
    # Ο δεύτερος πίνακας
    fig2 = plt.figure()   
    plt.plot(xtrain[:,0],xtrain[:,2],'b.')
    plt.plot(xtest[:,0],xtest[:,2],'r.')
    plt.title('2nd')   
    plt.show()
    
    # Ο τρίτος πίνακας, ομάδα 9 πινάκων
    fig3 = plt.figure()
    fig3.subplots_adjust(hspace=0.5, wspace=0.4)
    
    # Πίνακας απελπισίας [για χρησιμοποίηση ως άξονας x στα subplots]
    varethika = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.])
    predict = np.ones((15,1))
    
    # Αρχικοποιήσεις
    # [criterion]Ν -> Number, Άθροισμα αποτελεσμάτων
    accuracyN = 0.0
    precisionN = 0.0
    recallN = 0.0
    fmeasureN = 0.0
    sensitivityN = 0.0
    specificityN = 0.0
    # [criterion]Q -> Quantity, Πλήθος αποτελεσμάτων
    accuracyQ = 0.0
    precisionQ = 0.0
    recallQ = 0.0
    fmeasureQ = 0.0
    sensitivityQ = 0.0
    specificityQ = 0.0
    
    for folds in range(9):      
        xtrain, xtest, ttrain, ttest = train_test_split(
                x, t, test_size = 0.1) 
        
        ttest1 = np.copy(ttest)
        ttrain1 = np.copy(ttrain)
        
        # Πλήθος προτύπων 
        ptrain = len(ttrain)
        ptest = len(ttest)

        for i in range(ptrain):
            if ttrain[i] == 0:
                ttrain1[i] = -1
        for i in range(ptest):
            if ttest[i] == 0:
                ttest1[i] = -1

        
        # Διάνυσμα βαρών = διάνυσμα τροποποιημένων στόχων t (-1/1) 
        #                  * ψευδο-αντίστροδου πίνακα επαυξημένων προτύπων xtrain
        w = np.transpose(ttrain1).dot(np.linalg.pinv(np.transpose(xtrain)))
        # Το τελευταίο στοιχείο στην λύστα είναι η πόλωση (w0)

        # Έξοδος ταξινομητή για όλα τα πρότυπα του test set
        y = w.dot(np.transpose(xtest))

        # Εκτίμηση του ταξινομητή για την κλάση που ανήκουν τα πρότυπα του test set
        for i in range(15):
            if y[0,i] < 0:
                predict[i] = 0
            else:
                predict[i] = 1

        # Διαγράμματα
        ax = fig3.add_subplot(3,3,folds+1)
        ax.plot(varethika,ttest[:,0], 'b.')
        ax.plot(varethika, predict[:,0], 'r+') # Ζητάει η άσκηση κόκκινους κύκλους ('ro'), αλλά νομίζω το '+' είναι πιο ευδιάκριτο
        ax.set_title('Fold %i' %folds)
        
        # Αθροίσματα
        accuracyN, accuracyQ = countEval(accuracyN,accuracyQ,ttest,predict,'accuracy')
        precisionN, precisionQ = countEval(precisionN,precisionQ,ttest,predict,'precision')
        recallN, recallQ = countEval(recallN,recallQ,ttest,predict,'recall')
        fmeasureN, fmeasureQ = countEval(fmeasureN,fmeasureQ,ttest,predict,'fmeasure')
        sensitivityN, sensitivityQ = countEval(sensitivityN,sensitivityQ,ttest,predict,'sensitivity')
        specificityN, specificityQ = countEval(specificityN,specificityQ,ttest,predict,'specificity')
        
    plt.show()  
    
    
    print(accuracyQ)
    print(precisionQ)
    print(recallQ)
    print(fmeasureQ)
    print(sensitivityQ)
    print(specificityQ)
    # Μέση Τιμή
    print('\033[31m' + '-------MT-------'+ '\033[0m')
    print('\033[32m' + '~Accuracy:'+ '\033[0m')
    print(accuracyN/accuracyQ)
    print('\033[32m' + '~Precision:'+ '\033[0m')
    print(precisionN/precisionQ)
    print('\033[32m' + '~Recall:'+ '\033[0m')
    print(recallN/recallQ)
    print('\033[32m' + '~Fmeasure:'+ '\033[0m')
    print(fmeasureN/fmeasureQ)
    print('\033[32m' + '~Sensitivity:'+ '\033[0m')
    print(sensitivityN/sensitivityQ)
    print('\033[32m' + '~Specificity:'+ '\033[0m')
    print(specificityN/specificityQ)
    print('\033[31m' + '----------------')
    
    ans = raw_input('Do you want to continue? [type "y" for "yes", anything else for no]: '+ '\033[0m')

print('\n------------------------------THE END------------------------------')
''' -------------------- END MAIN -------------------- '''





