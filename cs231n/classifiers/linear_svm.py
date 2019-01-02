import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    loss_contributor_counts = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
    
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:] # update for incorrect class 
        loss_contributor_counts += 1
    dW[:,y[i]] += (-1) * (loss_contributor_counts) * X[i,:]
    
    
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = dW / num_train
  dW += reg*2*W
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
#   dW /= num_train
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = scores[:,y]
#   print('correct sc', correct_scores,correct_scores.shape)
  # subtract s_yi from all after computing all s_yi
  correct_scores = np.diagonal(correct_scores)
  correct_scores = np.reshape(correct_scores,(correct_scores.shape[0],-1))

  scores += 1 - correct_scores
  # make sure correct scores themselves don't contribute to loss function
  scores[list(range(num_train)), y] = 0
    
  loss = np.sum(np.sum(np.fmax(scores,0),axis=1),axis=0) / num_train
  # account for regularization
  loss = loss + reg * np.sum(W*W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
#   for i in range(num_train):
#     print(dW.shape)
#     dW += X[i,:][:,np.newaxis]
#     count = scores[i,scores[i,:]!=0].shape[0]
#     dW[list(range(dW.shape[0])),y[i]] -= count * X[i,:]
#     
#   dW = dW/num_train
  X_mask = np.zeros(scores.shape)
  X_mask[scores > 0] = 1
  X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
  dW = X.T.dot(X_mask)
  dW /= num_train
  dW += 2 * reg * W  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
