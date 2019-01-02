import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  import math
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  # exp_sj_sum = 0.0
  # print('X: ',X,'\n','W:',W,'\n')
  for i in range(num_samples):
    X_W = X[i,:].dot(W)  # calculate scores for all the classes at once
    max_term = np.max(X_W) # -log C
    # logC = -max_term
    s_yi = X_W[y[i]]-max_term
    exp_s_yi = np.exp(s_yi)
    # print('exp_s_yi shape:',np.exp(X_W-max_term).shape)
    exp_sj_all = np.exp(X_W-max_term)
    exp_sj_sum = np.sum(exp_sj_all)
    loss -= math.log(exp_s_yi/exp_sj_sum)
    dW[:,y[i]] += (1 - exp_s_yi/exp_sj_sum) * X[i,:]
    for j in range(num_classes):
      if (j == y[i]):
        continue
      else:
        dW[:,j] -= (1/(exp_sj_sum)) * exp_sj_all[j] * X[i,:]
  loss /= num_samples
  reg_loss = reg*np.sum(W*W)
  # print('reg_loss: ',reg_loss)
  loss += reg_loss
  dW = dW / (-num_samples)
  dW += 2*reg*W
  # print('total loss: ',loss)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = X.shape[0]
  num_classes = W.shape[1]
  
  # loss
  scores = X.dot(W)
  max_scores = np.max(scores,axis=1)
  max_scores = np.reshape(max_scores,(max_scores.shape[0],-1))
  scores -= max_scores
  correct_scores = scores[list(range(y.shape[0])),y]
  exp_scores_all = np.exp(scores)
  exp_correct_scores = np.exp(correct_scores)
  prob_ratio = exp_correct_scores/np.sum(exp_scores_all,axis=1)
  log_ratio = (-1)*np.log(prob_ratio)
  # print(log_ratio.shape)
  # print(np.sum(log_ratio,axis=0)
  loss = np.sum(log_ratio,axis=0) * (1/num_samples) + reg * np.sum(W*W)

  # gradient
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

