import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

def predictAccuracy(H, y):
    prediction = tf.equal(tf.argmax(H,axis=1), tf.argmax(y,axis = 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    
    return prediction, accuracy

def getOptimizer(late):
    optimizer = tf.train.AdamOptimizer(late)
    train = optimizer.minimize(cost)
    return train

def getCost(mylogist, mylabel):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=mylogist, labels=mylabel)
    cost = tf.reduce_mean(diff)
    return cost


mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

img_row = 28
img_column = 28
mnistImg = img_row * img_column
nb_classes = 10

x = tf.placeholder(tf.float32, [None, mnistImg])
y = tf.placeholder(tf.float32, [None, nb_classes])
###########################################################################
w1 = tf.Variable(tf.random_normal([mnistImg, 256]))
w1 = tf.get_variable('w1', shape=[mnistImg,256],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
H1 = tf.nn.relu(tf.matmul(x, w1) + b1) 
###########################################################################
w2 = tf.Variable(tf.random_normal([256, 256]))
w2 = tf.get_variable('w2', shape=[256,256],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
H2 = tf.nn.relu(tf.matmul(H1, w2) + b2) 
###########################################################################
w3 = tf.Variable(tf.random_normal([256, nb_classes]))
w3 = tf.get_variable('w3', shape=[256,nb_classes],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(H2, w3) + b3
H = tf.nn.softmax(logits) 
###########################################################################
cost = getCost(logits,y)

learn_rate = 0.001
train = getOptimizer(learn_rate)

#prediction = tf.equal(tf.argmax(H, axis=1) , tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

training_epochs = 15
batch_size = 100
total_example = mnist.train._num_examples
total_batch = int(total_example/ batch_size)
print(total_example)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        feed_data = {x:batch_xs,y:batch_ys}
        _cost, _train = sess.run([cost, train], feed_dict=feed_data)
        
        total_cost += _cost
        
    avg_cost = total_cost / total_batch # 평균비용
    print('epoch : %04d, cost(비용) : %.9f' % ( epoch+1, avg_cost )) 

print('학습종료')

prediction, accuracy = predictAccuracy(H, y)

print('정확도 : ', end =' ' )


feed_data = {x:mnist.test.images ,y:mnist.test.labels}
print(sess.run([accuracy], feed_dict= feed_data))


randItem = random.randint(0, mnist.test.num_examples - 1)
print('라벨 보기 : ', end=' ')
print(sess.run(tf.argmax(mnist.test.labels[randItem:randItem + 1] , axis=1)))

print('예측된  데이터 : ', end=' ')
print(sess.run( tf.argmax(H, axis=1), feed_dict={x:mnist.test.images[randItem:randItem + 1]}))
    
    
plt.imshow(
    mnist.test.images[randItem: randItem + 1].reshape(img_row, img_column),
    cmap = 'Greys',
    interpolation = 'nearest'
    )


plt.show()