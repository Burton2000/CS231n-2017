from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################

    x_Wx = np.dot(x, Wx)
    prev_h_Wh = np.dot(prev_h, Wh)

    # Calculate next hidden state = tanh(x*Wx + prev_H*Wh + b)
    next_h = np.tanh(x_Wx + prev_h_Wh + b)

    cache = (Wx, Wh, b, x, prev_h, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    # Grab our cached variables needed for backprop.
    Wx, Wh, b, x, prev_h, next_h = cache

    # Backprop dnext_h through the tanh function first, derivative of tanh is 1-tanh^2.
    dtanh = (1 - np.square(next_h)) * dnext_h

    # Backprop dtanh to calculate the other derivates (no complicated derivatives here).
    db = np.sum(dtanh, axis=0)
    dWh = np.dot(prev_h.T, dtanh)
    dprev_h = np.dot(dtanh, Wh.T)
    dWx = np.dot(x.T, dtanh)
    dx = np.dot(dtanh, Wx.T)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################

    N, T, D = x.shape
    N, H = h0.shape

    # Initialise h for holding our calculated hidden states.
    h = np.zeros([N, T, H])
    cache = []

    # Our initial hidden state
    prev_h = h0

    # RNN forward for T time steps.
    for t_step in range(T):
        cur_x = x[:, t_step, :]
        prev_h, cache_temp = rnn_step_forward(cur_x, prev_h, Wx, Wh, b)
        h[:, t_step, :] = prev_h
        cache.append(cache_temp)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################

    Wx, Wh, b, x, prev_h, next_h = cache[0]
    N, T, H = dh.shape
    D, H = Wx.shape

    # Initialise gradients.
    dx = np.zeros([N, T, D])
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dprev_h = np.zeros_like(prev_h)

    # Backprop in time - start at last calculated time step and work back.
    for t_step in reversed(range(T)):

        # Add the current timestep upstream gradient to previous calculated dh
        cur_dh = dprev_h + dh[:,t_step,:]

        # Calculate gradients at this time step.
        dx[:, t_step, :], dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(cur_dh, cache[t_step])

        # Add gradient contributions from each time step.
        dWx += dWx_temp
        dWh += dWh_temp
        db += db_temp

    # dh0 is the last hidden state gradient calculated.
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################

    # For each element in x we get its corresponding word embedding vector from W.
    out = W[x, :]

    cache = x, W
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html   #                                    #
    ##############################################################################

    # Get caches variables and initialise dW.
    x, W = cache
    dW = np.zeros_like(W)

    # Adds the upcoming gradients into the corresponding index from W.
    np.add.at(dW, x, dout)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################

    # Get H for slicing up activation vector A.
    H = np.shape(prev_h)[1]

    # Calculate activation vector A=x*Wx + prev_h*Wh + b.
    a_vector = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

    # Slice activation vector to get the 4 parts of it: input/forget/output/block.
    a_i = a_vector[:, 0:H]
    a_f = a_vector[:, H:2*H]
    a_o = a_vector[:, 2*H:3*H]
    a_g = a_vector[:, 3*H:]

    # Activation functions applied to our 4 gates.
    input_gate = sigmoid(a_i)
    forget_gate = sigmoid(a_f)
    output_gate = sigmoid(a_o)
    block_input = np.tanh(a_g)

    # Calculate next cell state.
    next_c = (forget_gate * prev_c) + (input_gate * block_input)

    # Calculate next hidden state.
    next_h = output_gate * np.tanh(next_c)

    # Cache variables needed for backprop.
    cache = (x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    # Reference for LSTM graph
    # http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/

    x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h = cache

    # Backprop dnext_h through the multiply gate.
    dh_tanh = dnext_h * output_gate
    da_o_partial = dnext_h * np.tanh(next_c)

    # Backprop dh_tanh through the tanh function.
    dtanh = dh_tanh * (1 - np.square(np.tanh(next_c)))

    # We add dnext_c and dtanh as during forward pass we split activation (gradients add up at forks).
    dtanh_dc = (dnext_c + dtanh)

    # Backprop dtanh_dc to calculate dprev_c.
    dprev_c = dtanh_dc * forget_gate

    # Backprop dtanh_dc towards each gate.
    da_i_partial = dtanh_dc * block_input
    da_g_partial = dtanh_dc * input_gate
    da_f_partial = dtanh_dc * prev_c

    # Backprop through gate activation functions to calculate gate derivatives.
    da_i = input_gate*(1-input_gate) * da_i_partial
    da_f = forget_gate*(1-forget_gate) * da_f_partial
    da_o = output_gate*(1-output_gate) * da_o_partial
    da_g = (1-np.square(block_input)) * da_g_partial

    # Concatenate back up our 4 gate derivatives to get da_vector.
    da_vector = np.concatenate((da_i, da_f, da_o, da_g), axis=1)

    # Backprop da_vector to get remaining gradients.
    db = np.sum(da_vector, axis=0)
    dx = np.dot(da_vector, Wx.T)
    dWx = np.dot(x.T, da_vector)
    dprev_h = np.dot(da_vector, Wh.T)
    dWh = np.dot(prev_h.T, da_vector)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################

    N, T, D = x.shape
    N, H = h0.shape

    cache = []
    h = np.zeros([N, T, H])

    # Set initial h and c states.
    prev_h = h0
    prev_c = np.zeros_like(h0)

    for time_step in range(T):
        prev_h, prev_c, cache_temp = lstm_step_forward(x[:,time_step,:], prev_h, prev_c, Wx, Wh, b)
        h[:, time_step, :] = prev_h  # Store the hidden state for this time step.
        cache.append(cache_temp)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################

    x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h = cache[0]

    N, T, H = dh.shape
    D, _ = Wx.shape

    dx = np.zeros([N, T, D])
    dprev_h = np.zeros_like(prev_h)
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    # Initial gradient for cell is all zero.
    dprev_c = np.zeros_like(dprev_h)

    for time_step in reversed(range(T)):

        # Add the current timestep upstream gradient to previous calculated dh.
        cur_dh = dprev_h + dh[:,time_step,:]

        dx[:,time_step,:], dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(cur_dh, dprev_c, cache[time_step])

        # Add gradient contributions from each time step together.
        db += db_temp
        dWh += dWh_temp
        dWx += dWx_temp

    # dh0 is the last hidden state gradient calculated.
    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
