from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    # Number of images in the batch.
    NN = x.shape[0]

    # Reshape each input in our batch to a vector.
    reshaped_input = np.reshape(x,[NN, -1])

    # FC layer forward pass.
    out = np.dot(reshaped_input, w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    # Number of images in the batch.
    NN = x.shape[0]

    # Reshape each input in our batch to a vector.
    reshaped_x = np.reshape(x,[NN, -1])

    # Calculate dx = w*dout - remember to reshape back to shape of x.
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)

    # Calculate dw = x*dout
    dw = np.dot(reshaped_x.T,dout)

    # Calculate db = dout
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    # Forward Relu.
    out = x.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    # For Relu we only backprop to non-negative elements of x
    relu_mask = (x >= 0)
    dx = dout * relu_mask

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        # Take sample mean & var of our minibatch across each dimension.
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        # Normalise our batch then shift and scale with gamma/beta.
        normalized_data = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * normalized_data + beta

        # Update our running mean and variance then store.
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        # Store intermediate results needed for backward pass.
        cache = {
            'x_minus_mean': (x - sample_mean),
            'normalized_data': normalized_data,
            'gamma': gamma,
            'ivar': 1./np.sqrt(sample_var + eps),
            'sqrtvar': np.sqrt(sample_var + eps),
        }

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        # Test time batch norm using learned gamma/beta and calculated running mean/var.
        out = (gamma / (np.sqrt(running_var + eps)) * x) + (beta - (gamma*running_mean)/np.sqrt(running_var + eps))

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    # Get cached results from the forward pass.
    N, D = dout.shape
    normalized_data = cache.get('normalized_data')
    gamma = cache.get('gamma')
    ivar = cache.get('ivar')
    x_minus_mean = cache.get('x_minus_mean')
    sqrtvar = cache.get('sqrtvar')

    # Backprop dout to calculate dbeta and dgamma.
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * normalized_data, axis=0)

    # Carry on the backprop in steps to calculate dx.
    # Step1
    dxhat = dout*gamma
    # Step2
    dxmu1 = dxhat*ivar
    # Step3
    divar = np.sum(dxhat*x_minus_mean, axis=0)
    # Step4
    dsqrtvar = divar * (-1/sqrtvar**2)
    # Step5
    dvar = dsqrtvar * 0.5 * (1/sqrtvar)
    # Step6
    dsq = (1/N)*dvar*np.ones_like(dout)
    # Step7
    dxmu2 = dsq * 2 * x_minus_mean
    # Step8
    dx1 = dxmu1 + dxmu2
    dmu = -1*np.sum(dxmu1 + dxmu2, axis=0)
    # Step9
    dx2 = (1/N)*dmu*np.ones_like(dout)
    # Step10
    dx = dx2 + dx1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    # Get cached variables from foward pass.
    N, D = dout.shape
    normalized_data = cache.get('normalized_data')
    gamma = cache.get('gamma')
    ivar = cache.get('ivar')
    x_minus_mean = cache.get('x_minus_mean')
    sqrtvar = cache.get('sqrtvar')

    # Backprop dout to calculate dbeta and dgamma.
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * normalized_data, axis=0)

    # Alternative faster formula way of calculating dx. ref: http://cthorey.github.io./backpropagation/
    dx =(1 / N) * gamma * 1/sqrtvar * ((N * dout) - np.sum(dout, axis=0) - (x_minus_mean) * np.square(ivar) * np.sum(dout * (x_minus_mean), axis=0))

###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        # Ref on dropout: http://cs231n.github.io/neural-networks-2/

        # During training randomly drop out neurons with probability P, here we create the mask that does this.
        mask = (np.random.random_sample(x.shape) >= p)

        # Inverted dropout scales the remaining neurons during training so we don't have to at test time.
        dropout_scale_factor = 1/(1-p)
        mask = mask*dropout_scale_factor

        # Apply the dropout mask to the input.
        out = x*mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        # Test time we don't drop anything so just pass input through, also scaling was done during training.
        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        # Only backprop to the neurons we didn't drop.
        dx = dout*mask*1

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Grab conv parameters
    pad = conv_param.get('pad')
    stride = conv_param.get('stride')

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # Zero pad our tensor along the spatial dimensions.
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))

    # Calculate output spatial dimensions.
    out_H = np.int(((H + 2*pad - HH) / stride) + 1)
    out_W = np.int(((W + 2*pad - WW) / stride) + 1)

    # Initialise the output.
    out = np.zeros([N, F, out_H, out_W])

    # Naive convolution loop.
    for nn in range(N):  # For each image in the input batch.
        for ff in range(F):  # For each filter in w
            for jj in range(0, out_H):  # For each output pixel height
                for ii in range(0, out_W):  # For each output pixel width
                    out[nn, ff, jj, ii] = \
                        np.sum(w[ff, ...] * padded_x[nn, :, jj*stride:jj*stride+HH, ii*stride:ii*stride+WW]) + b[ff]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    # Grab conv parameters and pad x if needed.
    x, w, b, conv_param = cache
    stride = conv_param.get('stride')
    pad = conv_param.get('pad')
    padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape

    # Initialise gradient output tensors.
    dx_temp = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Calculate dB.
    # Just like in the affine layer we sum up all the incoming gradients for each filters bias.
    for ff in range(F):
        db[ff] += np.sum(dout[:, ff, :, :])

    # Calculate dw.
    # By chain rule dw is dout*x
    for nn in range(N):
        for ff in range(F):
            for jj in range(H_out):
                for ii in range(W_out):
                    dw[ff, ...] += dout[nn, ff, jj, ii] * padded_x[nn,:,jj*stride:jj*stride+HH,ii*stride:ii*stride+WW]

    # Calculate dx.
    # By chain rule dx is dout*w. We need to make dx same shape as padded x for the gradient calculation.
    for nn in range(N):
        for ff in range(F):
            for jj in range(H_out):
                for ii in range(W_out):
                    dx_temp[nn, :, jj*stride:jj*stride+HH,ii*stride:ii*stride+WW] += dout[nn, ff, jj,ii] * w[ff, ...]

    # Remove the padding from dx so it matches the shape of x.
    dx = dx_temp[:, :, pad:H+pad, pad:W+pad]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    # Grab the pooling parameters.
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    N, C, H, W = x.shape

    # Calculate output spatial dimensions.
    out_H = np.int(((H - pool_height) / stride) + 1)
    out_W = np.int(((W - pool_width) / stride) + 1)

    # Initialise output.
    out = np.zeros([N, C, out_H, out_W])

    # Naive maxpool for loop.
    for n in range(N):  # For each image.
        for c in range(C):  # For each channel
            for j in range(out_H):  # For each output row.
                for i in range(out_W):  # For each output col.
                    out[n, c, j, i] = np.max(x[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################

    # Grab the pooling parameters.
    x, pool_param = cache
    pool_height = pool_param.get('pool_height')
    pool_width = pool_param.get('pool_width')
    stride = pool_param.get('stride')

    N, C, H, W = x.shape
    _, _, dout_H, dout_W = dout.shape

    # Initialise dx to be same shape as maxpool input x.
    dx = np.zeros_like(x)

    # Naive loop to backprop dout through maxpool layer.
    for n in range(N):  # For each image.
        for c in range(C):  # For each channel
            for j in range(dout_H):  # For each row of dout.
                for i in range(dout_W):  # For each col of dout.

                    # Using argmax get the linear index of the max of each patch.
                    max_index = np.argmax(x[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width])
                    # Using unravel_index convert this linear index to matrix coordinate.
                    max_coord = np.unravel_index(max_index, [pool_height,pool_width])
                    # Only backprop the dout to the max location.
                    dx[n,c, j*stride:j*stride+pool_height, i*stride:i*stride+pool_width][max_coord] = dout[n, c, j, i]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    # We transpose to get channels as last dim and then reshape to size (-1, C) so we can use normal batchnorm.
    x_t = x.transpose((0, 2, 3, 1))
    x_flat = x_t.reshape(-1, x.shape[1])

    out, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    # Reshape our results back to our desired shape.
    out_reshaped = out.reshape(*x_t.shape)
    out = out_reshaped.transpose((0, 3, 1, 2))


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    # We transpose to get channels as last dim and then reshape to size (-1, C) so we can use normal batchnorm.
    dout_t = dout.transpose((0, 2, 3, 1))
    dout_flat = dout_t.reshape(-1, dout.shape[1])

    dx, dgamma, dbeta = batchnorm_backward(dout_flat, cache)

    # We need to reshape dx back to our desired shape.
    dx_reshaped = dx.reshape(*dout_t.shape)
    dx = dx_reshaped.transpose((0, 3, 1, 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
