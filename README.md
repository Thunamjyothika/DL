A feedforward neural network is a type of artificial neural network where data flows in one direction.
The FNN contail mainly three components they are input layer, hidden layer, output layer.
Load the CIFAR-10 dataset and divided into training and test sets.
In this code we kept the learning rate it control the step size during gradient updates.
Batch size it defines how many samples are processed before the model's weights are updated.
Hidden size specifies the number of neurons in the hidden layer.
Epochs defines the number of times the entire dataset will be processed during training.
The weights for both layers are initialized randomly using a small scale and bias are initialized as zeros.
I used two acctivation functions sigmoid function for hidden layer introduce non-linearity to the model,softmax function converts the output into propabilities.
Input is passed through the hidden layer and then through the output layer.The output is compared with the true label using cross-entropy loss.
For the backpropagation I used the gradient it uses the chain rule.
The network is trained over multiple epochs, and the loss id printed every 10 epochs.
After training the test set is passed through the network.
The output is obtained by passing test data through the hidden layer,followed by the output layer.
The prediction are made using the softmax function, and accuracy is computed to evaluate the model's performance on unseen data.
