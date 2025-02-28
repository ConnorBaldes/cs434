import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import logging
import time
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.015
max_iters=1000

def main():
  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()
  # data to be plotted

  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  for k in [4,5]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  #raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')
  w, losses = trainLogistic(X_train_bias, y_train)
    #x_pred_test = X_test@w >= 0
  r = open("results.csv", "a")
  r.write("id,type\n")
  for x, point in enumerate(X_test):
    x_pred_test = np.dot(X_test[x],w[1:])
    pred = (np.sum(x_pred_test)+w[0])
    if pred >= 0:
      line = str(x)+","+str(1)+"\n"   
      r.write(line)
    else:
      line = str(x)+","+str(0)+"\n"   
      r.write(line)
      
  r.close()

def dummyAugment(X):
      
  bias = np.ones((X.shape[0],1))
  new_x = np.hstack((bias,X))
  return new_x
  #raise Exception('Student error: You haven\'t implemented dummyAugment yet.')



def calculateNegativeLogLikelihood(X,y,w):
      

  neglog = 0
  sum = 0
  const = 0.0000001
  for idx, point in enumerate(X):
    prd1 = 1/(1+np.exp(-1*(np.dot(np.transpose(w)[0],X[idx]))))
    prd2 = 1-prd1
    sum += (y[idx][0]*(np.log(prd1+const)) + ((1-y[idx][0])*(np.log(prd2+const))))
  neglog = (-1)*sum

  return neglog

  #raise Exception('Student error: You haven\'t implemented the negative log likelihood calculation yet.')
 

def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )

    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
    
        # Make a variable to store our gradient
        w_grad = np.zeros( (X.shape[1],1 ))

        # Compute the gradient over the dataset and store in w_grad
        # .
        # . Implement equation 9.
        # .
        #print(np.transpose(w))
        prd = 0
        for idx, point in enumerate(X):

              prd = 1/(1+np.exp(-1*(np.dot(np.transpose(w)[0],X[idx]))))
              grd = np.array([(prd - y[idx][0])*X[idx]])

              w_grad = w_grad + grd.transpose()

        #raise Exception('Student error: You haven\'t implemented the gradient calculation for trainLogistic yet.')

        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad
        # Calculate the negative log-likelihood with the 
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))
        
    return w, losses




##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
