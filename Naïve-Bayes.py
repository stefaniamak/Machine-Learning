#NAME: STEFANIA MAKRYGIANNAKI
#AM: 164703
#ΤΜΗΜΑ: T3

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm

''' -------------------- Συναρτήσεις -------------------- '''

def nbtrain( x, t ):
# Είσοδος x : Pxn πίνακας με τα πρότυπα (P=πλήθος προτύπων, n=διάσταση)
# Είσοδος t : διάνυσμα με τους στόχους (0/1)
# Έξοδος model : dictionary που θα περιέχει τις παραμέτρους του μοντέλου
    
    # Χωρισμός προτύπων στην κλάση 0 και στην κλάση 1 
    xClass1 = []
    xClass0 = []
    
    xlen = len(x)
    for i in range(xlen):
        if t[i] == 1:
            xClass1.append(x[i,:])
        else:
            xClass0.append(x[i,:])
    
    # Πλήθος προτύπων σε κάθε κλάση
    x0_len = len(xClass0)
    x1_len = len(xClass1)     

    # Αρχικοποίηση μέσης τιμής "μ" και διασποράς "σ"
    mu = np.zeros((2,4))
    sigma = np.zeros((2,4))
    
    prior = np.array([(x0_len+0.0) / (xlen+0.0), (x1_len+0.0) / (xlen+0.0)])
    
    mu[0] = np.mean(xClass0, axis = 0) # μέση τιμή του κάθε χαρακτηριστικού για την κλαση 0
    mu[1] = np.mean(xClass1, axis = 0) # διασπορά του κάθε χαρακτηριστικού για την κλαση 0
    sigma[0]= np.std(xClass0, axis = 0) # μέση τιμή του κάθε χαρακτηριστικού για την κλαση 1
    sigma[1]= np.std(xClass1, axis = 0) # διασπορά του κάθε χαρακτηριστικού για την κλαση 1

    # Επιστροφή του dictionary 
    return {"prior": prior,
            "mu": mu,
            "sigma": sigma}


def nbpredict( x, model ):
# Είσοδος x : Pxn πίνακας με τα πρότυπα
# Είσοδος model : dictionary με τις παραμέτρους του μοντέλου NB 
# Έξοδος predict : διάνυσμα με τις εκτιμώμενες τιμές στόχου 
    
    G = norm.pdf # norm.pdf(x, loc=μ, scale=σ)
    mu = model["mu"]
    sigma = model["sigma"]
    predictTest = np.zeros((len(x))).astype(float) 
    
    for p in range(len(x)):   
        # Υπολογισμός του λόγου των πιθανοτήτν L = prior[1] / prior[0]
        L = (model["prior"][0] / model["prior"][1])
        for i in range(4):
            L = L*(G(x[p][i], mu[1][i], sigma[1][i]) / G(x[p][i], mu[0][i], sigma[0][i]))
        if L > 1.0:
            predictTest[p] = 1.0
            
    return predictTest


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
    
    # Δημιουργία πινάκων xtrain, xtest, ttrain, ttest
    xtrain = np.vstack((x[0:40],x[50:90],x[100:140])).astype(float)  
    xtest = np.vstack((x[40:50],x[90:100],x[140:150])).astype(float)    
    ttrain = np.vstack((t[0:40],t[50:90],t[100:140])).astype(float)  
    ttest = np.vstack((t[40:50],t[90:100],t[140:150])).astype(float) 
    
    # δεύτερος πίνακας
    fig2 = plt.figure()   
    plt.plot(xtrain[:,0],xtrain[:,2],'b.')
    plt.plot(xtest[:,0],xtest[:,2],'r.')
    plt.title('2nd')   
    plt.show()
    
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
    
    predict = np.ones((15,1))
    
    # Ο τέταρτος πίνακας, ομάδα 9 πινάκων
    fig3 = plt.figure()
    fig3.subplots_adjust(hspace=0.5, wspace=0.4)
    
    '''FOLDS'''    
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
  
    
        # Εκπαίδευση μοντέλου Naive Bayes κάνοντας την υπόθεση ότι τα χαρακτηριστικά ακολουθούν την Γκαουσσιανή κατανομή
        model = nbtrain(xtrain,ttrain)
        # Ανάκληση
        predictTest = nbpredict(xtest,model)
        
        # Διέγερση νευρώνα για όλα τα πρότυπα του test set
        for i in range(ptest):   
            # Εκτίμηση του ταξινομητή για την κλάση που ανήκουν τα πρότυπα του test set
            if predictTest[i] < 0.5:
                predict[i] = 0
            else:
                predict[i] = 1

        # Αθροίσματα
        accuracyN, accuracyQ = countEval(accuracyN,accuracyQ,ttest,predict,'accuracy')
        precisionN, precisionQ = countEval(precisionN,precisionQ,ttest,predict,'precision')
        recallN, recallQ = countEval(recallN,recallQ,ttest,predict,'recall')
        fmeasureN, fmeasureQ = countEval(fmeasureN,fmeasureQ,ttest,predict,'fmeasure')
        sensitivityN, sensitivityQ = countEval(sensitivityN,sensitivityQ,ttest,predict,'sensitivity')
        specificityN, specificityQ = countEval(specificityN,specificityQ,ttest,predict,'specificity')
        
        # Διαγράμματα
        ax = fig3.add_subplot(3,3,folds+1)
        ax.plot(np.arange(15),ttest[:,0], 'b.')
        ax.plot(np.arange(15), predict[:,0], 'r+') # Ζητάει η άσκηση κόκκινους κύκλους ('ro'), αλλά νομίζω το '+' είναι πιο ευδιάκριτο
        ax.set_title('Fold %i' %folds)
        
    plt.show()  
            
    # Μέση Τιμή
    print('\033[31m' + '-------MT-------'+ '\033[0m')
    print('\033[32m' + '~Accuracy:'+ '\033[0m')
    print('%i%%' %(accuracyN/accuracyQ*100))
    print('\033[32m' + '~Precision:'+ '\033[0m')
    if precisionQ == 0:
        print('No positives found')
    else:
        print('%i%%' %(precisionN/precisionQ*100))
        print('\033[32m' + '~Recall:'+ '\033[0m')
        print('%i%%' %(recallN/recallQ*100))
        print('\033[32m' + '~Fmeasure:'+ '\033[0m')
        print('%i%%' %(fmeasureN/fmeasureQ*100))
        print('\033[32m' + '~Sensitivity:'+ '\033[0m')
        print('%i%%' %(sensitivityN/sensitivityQ*100))
        print('\033[32m' + '~Specificity:'+ '\033[0m')
        print('%i%%' %(specificityN/specificityQ*100))
        print('\033[31m' + '----------------')
    

    ans = raw_input('Do you want to continue? [type "y" for "yes", anything else for no]: '+ '\033[0m')

print('\n------------------------------THE END-------------------------------')

''' -------------------- END MAIN -------------------- '''





