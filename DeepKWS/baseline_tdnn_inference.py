import tensorflow as tf

LAYER_NODE = 128
keep_prob = 0.9
OUTPUT_NODE = 2

def inference(input_tensor , maxT , channels , fbank_dim , training):

    x = tf.layers.conv1d(input_tensor , LAYER_NODE , 5 , padding = 'same' , dilation_rate = 1)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 5 , padding = 'same' , dilation_rate = 2)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 5 , padding = 'same' , dilation_rate = 4)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 5 , padding = 'same' , dilation_rate = 8)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 3 , padding = 'same' , dilation_rate = 1)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 3 , padding = 'same' , dilation_rate = 2)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 3 , padding = 'same' , dilation_rate = 4)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x , LAYER_NODE , 3 , padding = 'same' , dilation_rate = 8)
    x = tf.contrib.layers.batch_norm(x , is_training = training)
    x = tf.nn.relu(x)


    x = tf.nn.dropout(x , keep_prob = keep_prob)
    output = tf.layers.dense(x , OUTPUT_NODE)
    
    return output
