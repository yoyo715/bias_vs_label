# tensorflow_version.py

# This is supposed to recreate fasttext in order to check my implementation


from dictionary import Dictionary
import tensorflow as tf



def main():
    training_epochs = 20
    n_neurons_in_h1 = 30
    n_neurons_in_h2 = 30
    learning_rate = 0.01
    
    X = tf.placeholder(tf.float32, [None, n_features], name='features')
    Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
    
    W1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_features)), name='weights1')
    
    #network parameters(weights and biases) are set and initialized(Layer2)
    W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_features)),name='weights2')
    
    #output layer weights and biasies
    Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='weightsOut')
   
    #activation function(softmax)
    a = tf.nn.softmax((tf.matmul(y2, Wo)), name='activationOutputLayer')
    
    #cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a),reduction_indices=[1]))
    
    #optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    #compare predicted value from network with the expected value/target
    correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
    
    #accuracy determination
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    
    
    # initialization of all variables
    initial = tf.global_variables_initializer()

    #creating a session
    with tf.Session() as sess:
        sess.run(initial)
        writer = tf.summary.FileWriter("/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/TSTrainResults")
        writer.add_graph(sess.graph)
        merged_summary = tf.summary.merge_all()
        
        # training loop over the number of epoches
        batchsize=10
        for epoch in range(training_epochs):
            for i in range(len(tr_features)):

                start=i
                end=i+batchsize
                x_batch=tr_features[start:end]
                y_batch=tr_labels[start:end]
                
                # feeding training data/examples
                sess.run(train_step, feed_dict={X:x_batch , Y:y_batch,keep_prob:0.5})
                i+=batchsize
                
            # feeding testing data to determine model accuracy
            y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_features,keep_prob:1.0})
            y_true = sess.run(tf.argmax(ts_labels, 1))
            summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_labels,keep_prob:1.0})
            # write results to summary file
            writer.add_summary(summary, epoch)
            # print accuracy for each epoch
            print('epoch',epoch, acc)
            print ('---------------')
            print(y_pred, y_true)


if __name__ == '__main__':
    main()
    
    
    
    
    