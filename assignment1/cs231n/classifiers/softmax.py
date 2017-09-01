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

  # Calculate loss and gradient for each element of our batch.
  for ii in range(num_train):
    current_scores = scores[ii, :]

    # Fix for numerical stability by subtracting max from score vector.
    shift_scores = current_scores - np.max(current_scores)

    # Calculate loss for this example.
    loss_ii = -shift_scores[y[ii]] + np.log(np.sum(np.exp(shift_scores)))
    loss += loss_ii

    for jj in range(num_classes):
      softmax_score = np.exp(shift_scores[jj]) / np.sum(np.exp(shift_scores))

      # Gradient calculation.
      if jj == y[ii]:
        dW[:, jj] += (-1 + softmax_score) * X[ii]
      else:
        dW[:, jj] += softmax_score * X[ii]

  # Average over the batch and add our regularization term.
  loss /= num_train
  loss += reg * np.sum(W*W)

  # Average over the batch and add derivative of regularization term.
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

  num_train = X.shape[0]

  # Calculate scores and numeric stability fix.
  scores = np.dot(X, W)
  shift_scores = scores - np.max(scores, axis=1)[...,np.newaxis]

  # Calculate softmax scores.
  softmax_scores = np.exp(shift_scores)/ np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]

  # Calculate dScore, the gradient wrt. softmax scores.
  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  # Backprop dScore to calculate dW, then average and add regularisation.
  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  # Calculate our cross entropy Loss.
  correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  loss = np.sum(loss)

  # Average our loss then add regularisation.
  loss /= num_train
  loss += reg * np.sum(W*W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

