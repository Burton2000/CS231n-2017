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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X, W)

  # calculate loss and gradient for each element of our batch
  for ii in range(num_train):
    current_scores = scores[ii, :]

    # fix for numerical stability by subtracting the max score from the scores vector
    shift_scores = current_scores - np.max(current_scores)

    # calculate loss
    loss_ii = -shift_scores[y[ii]] + np.log(np.sum(np.exp(shift_scores)))
    loss += loss_ii

    for jj in range(num_classes):
      softmax_score = np.exp(shift_scores[jj]) / np.sum(np.exp(shift_scores))

      # gradient calculation
      if jj == y[ii]:
        dW[:, jj] += (-1 + softmax_score) * X[ii]
      else:
        dW[:, jj] += softmax_score * X[ii]

  # average over the batch and add our regularization term
  loss /= num_train
  loss += reg * np.sum(W * W)

  # average over the batch and add derivative of regularization term
  dW /= num_train
  dW += 2*reg*W

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

