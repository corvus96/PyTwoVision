def conv_layer_output_dim(dimension, filter_size, p, s):
    """
    Inputs:
    dimension: it is the dimension of convolutional layer,
    that can be width or height
    filter_size: their spatial extent i.e: a filter with dimensions 
    5x5 will have a filter_size of 5
    p: the amount of zero padding 
    s: the stride
    Output: The next layer dimension that depends 
    of previous hyperparameters
    """
    return int(((dimension - filter_size + 2 * p)/s) + 1)

def max_polling_layer_output_dim(dimension, filter_size, s):
    """
    Inputs:
    dimension: it is the dimension of convolutional layer,
    that can be width or height
    filter_size: their spatial extent i.e: a filter with dimensions 
    5x5 will have a filter_size of 5
    s: the stride
    Output: The next layer dimension that depends 
    of previous hyperparameters
    """
    return int(((dimension - filter_size)/s) + 1)