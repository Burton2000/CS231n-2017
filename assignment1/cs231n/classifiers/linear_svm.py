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

  # Initialize loss and the gradient of W to zero.
  dW = np.zeros(W.shape)
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Compute the data loss and the gradient.
  for i in range(num_train):  # For each image in training.
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_classes_greater_margin = 0

    for j in range(num_classes):  # For each calculated class score for this image.

      # Skip if images target class, no loss computed for that case.
      if j == y[i]:
        continue

      # Calculate our margin, delta = 1
      margin = scores[j] - correct_class_score + 1

      # Only calculate loss and gradient if margin condition is violated.
      if margin > 0:
        num_classes_greater_margin += 1
        # Gradient for non correct class weight.
        dW[:, j] = dW[:, j] + X[i, :]
        loss += margin

    # Gradient for correct class weight.
    dW[:, y[i]] = dW[:, y[i]] - X[i, :]*num_classes_greater_margin

  # Average our data loss across the batch.
  loss /= num_train

  # Add regularization loss to the data loss.
  loss += reg * np.sum(W * W)

  # Average our gradient across the batch and add gradient of regularization term.
  dW = dW /num_train + 2*reg *W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = np.dot(X, W)

  correct_class_scores = np.choose(y, scores.T)  # np.choose uses y to select elements from scores.T

  # Need to remove correct class scores as we dont calculate loss/margin for those.
  mask = np.ones(scores.shape, dtype=bool)
  mask[range(scores.shape[0]), y] = False
  scores_ = scores[mask].reshape(scores.shape[0], scores.shape[1]-1)

  # Calculate our margins all at once.
  margin = scores_ - correct_class_scores[..., np.newaxis] + 1

  # Only add margin to our loss if it's greater than 0, let's make
  # negative margins =0 so they dont change our loss.
  margin[margin < 0] = 0

  # Average our data loss over the size of batch and add reg. term to the loss.
  loss = np.sum(margin) / num_train
  loss += reg * np.sum(W * W)

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

  original_margin = scores - correct_class_scores[...,np.newaxis] + 1

  # Mask to identiy where the margin is greater than 0 (all we care about for gradient).
  pos_margin_mask = (original_margin > 0).astype(float)

  # Count how many times >0 for each image but dont count correct class hence -1
  sum_margin = pos_margin_mask.sum(1) - 1

  # Make the correct class margin be negative total of how many > 0
  pos_margin_mask[range(pos_margin_mask.shape[0]), y] = -sum_margin

  # Now calculate our gradient.
  dW = np.dot(X.T, pos_margin_mask)

  # Average over batch and add regularisation derivative.
  dW = dW / num_train + 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
