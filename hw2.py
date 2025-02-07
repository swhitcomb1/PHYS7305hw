import numpy as np
import random
import matplotlib.pyplot as plt


numensembles=50000
sigma=1
n=2
num_hist_bins=100

#PART A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def randomnumber(sigma):
    num = np.random.normal(loc=0, scale=sigma)
    return num

def genarraygaussian(size, stdev):
    array = np.zeros((size, size), dtype='float64')
    for i in range(size):
        for j in range(size):
            array[i,j] = randomnumber(stdev)
    return array

def genensemblesgaussian(M, stdev, size):
    arr = np.zeros((M), dtype=object)
    importanteigenvaluelist =np.zeros((M))

    for num in range(M):
        array = genarraygaussian(size, stdev)
        
        arr[num] = array + array.T # makes it symmetric
        novertwoindex = int(np.floor(size/2)) # this is N/2
        
        eigvalues, eigvectors = np.linalg.eig(arr[num]) # gets the eigvalues and eigvectors
        idx = eigvalues.argsort()[::1]  # sorts the eigenvectors
        eigvalues = eigvalues[idx] # same
        
        importanteigenvaluelist[num] = eigvalues[novertwoindex]-eigvalues[novertwoindex-1] # calculates splitting
    return importanteigenvaluelist

# UN COMMENT NEXT SIX LINES TO RUN PART A 

#for N in [2,4,10]:
#    splittinglist = genensemblesgaussian(numensembles, sigma, N)
#    a=plt.hist(splittinglist / np.mean(splittinglist), bins=num_hist_bins, density=True)
#    plt.title(f'Part A, N={n}')
#    plt.show()
#    plt.close()

# END PART A
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# PART E
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def comparewithwigner(numdimensions):
    s = np.arange(0, 5, .01)
    plt.plot(s, (np.pi * s / 2 * np.exp(-1 * np.pi * s**2 / 4)))
    
    splittinglist = genensemblesgaussian(numensembles, sigma, numdimensions)
    plcholder=plt.hist(splittinglist / np.mean(splittinglist), bins=num_hist_bins, density=True)

    plt.title(f'N = {numdimensions}') 
    plt.xlabel('s')
    plt.ylabel('Probability')
    plt.show()
    plt.close()

# UN COMMENT NEXT TWO LINES TO RUN PART F

#for i in [2, 4, 10]:
#    comparewithwigner(i)

# END PART E
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# PART F
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def genarrayplusminusone(size):
    array = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            randomseed = np.random.rand()
            if randomseed>0.5:
                array[i,j] = 1
            if randomseed<=0.5:
                array[i,j] = -1
    return array

def genensemblespm(M, size):
    arr = np.zeros((M), dtype=object)
    importanteigenvaluelist =np.zeros((M))

    for num in range(M):
        array = genarrayplusminusone(size)
        
        arr[num] = array + array.T # makes it symmetric
        novertwoindex = int(np.floor(size/2)) # this is N/2
        
        eigvalues, eigvectors = np.linalg.eig(arr[num]) # gets the eigvalues and eigvectors
        idx = eigvalues.argsort()[::1]  # sorts the eigenvectors
        eigvalues = eigvalues[idx] # same
        
        importanteigenvaluelist[num] = eigvalues[novertwoindex]-eigvalues[novertwoindex-1] # calculates splitting
    return importanteigenvaluelist

def comparepmwithwigner(numdimensions):
    s = np.arange(0, 5, .01)
    plt.plot(s, (np.pi * s / 2 * np.exp(-1 * np.pi * s**2 / 4)), label='Wigners surmise')
    
    splittinglist = genensemblespm(numensembles, numdimensions)
    plcholder=plt.hist(splittinglist / np.mean(splittinglist), bins=num_hist_bins, density=True, label='histogram')
    
    plt.xlim(0,5)
    plt.title(f'N = {numdimensions} plus minus one array') 
    plt.xlabel('s')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.legend()

    plt.show()
    plt.close()

def comparegaussian(N):
    splittinglist = genensemblespm(numensembles, N)
    a=plt.hist(splittinglist / np.mean(splittinglist), bins=num_hist_bins, density=True)
    s = np.arange(0, 5, .01)
    plt.plot(s, (np.pi * s / 2 * np.exp(-1 * np.pi * s**2 / 4)))
    plt.title(f'N= {N} plus minus one arrays')
    plt.ylabel('Probability')
    plt.yticks([])
    plt.xlabel('Splitting')
    plt.xlim(0,5)
    plt.show()
    plt.close()
    
    splittinglist = genensemblesgaussian(numensembles, sigma, N)
    a=plt.hist(splittinglist / np.mean(splittinglist), bins=num_hist_bins, density=True)
    s = np.arange(0, 5, .01)
    plt.plot(s, (np.pi * s / 2 * np.exp(-1 * np.pi * s**2 / 4)))
    plt.title(f'N= {N} gaussian arrays')
    plt.ylabel('Probability')
    plt.xlabel('Splitting')
    plt.show()
    plt.close()

#comparegaussian(10)

for i in [2, 4, 10]:
    comparepmwithwigner(i)
