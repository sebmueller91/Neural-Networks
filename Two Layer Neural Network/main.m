function main
  % Load test data
  images = loadMNISTImages('MNIST Dataset/train-images.idx3-ubyte');
  labels = loadMNISTLabels('MNIST Dataset/train-labels.idx1-ubyte')';
  
  % Split data in training and test set
  images_train = images(:,1:50000);
  labels_train = labels(:,1:50000);
  
  images_test = images(:,50001:60000);
  labels_test = labels(:,50001:60000);
  
  %labels(1:5,1)
  %img = reshape(images(:,5), [28,28]) .* 255;
  %image(img)
  
  % declare parameters   
  s0 =784; s1 = 500; s2 = 10;
  
  % Train the network with the given parameters
  [w1,b1,w2,b2] = TrainTwoLayerNN(s0,s1,s2, images_train, labels_train);
  
  % Use the network to predict the test images
  [z1,z2,a1,a2] = feedforward(w1,b1,w2,b2,images_test);
  
  % Only take the maximum number in each row
  [max_val, predicted_labels] = max(a2);
  predicted_labels = predicted_labels - ones(size(predicted_labels)); % idx 1 corresponds to zero and so on
  
  % Count the number of entries where the predicted label is equal to the test label
  numberCorrectlyClassified = sum(predicted_labels == labels_test);
  
  % Print the success rate to the console
  fprintf("Correctly classified: %d/%d", numberCorrectlyClassified, size(labels_test,2));
end

function [w1,b1,w2,b2] = TrainTwoLayerNN(s0,s1,s2, train_input, train_output)
  % variables and parameters
  eps = 0.01;
  grad_norm = inf;
  eta = 1;
  batch_size = 200;
  
  % initialize weights and biases randomly
  w1 = randn([s1,s0]); 
  b1 = ones(s1,1);
  w2 = randn([s2,s1]);
  b2 = ones(s2,1);
  
  % Stochastic Gradient Descent
  while (grad_norm > eps)
    % Create batches 
    [input_batch, output_batch] = CreateBatch(batch_size, train_input, train_output);
    
    % Calculate derivative with backpropagation
    [w1_grad, b1_grad, w2_grad, b2_grad] = backpropagation(w1,b1,w2,b2,input_batch,output_batch);
    
    % Do one step in the direction of the derivative
    w1 = w1 - (eta/batch_size).*w1_grad;
    b1 = b1 - (eta/batch_size).*b1_grad;
    w2 = w2 - (eta/batch_size).*w2_grad;
    b1 = b2 - (eta/batch_size).*b2_grad;
    
    % Calculate the new norm of the gradient
    grad_norm = norm(w1) + norm(w2);
  end
end

function [w1_grad, b1_grad, w2_grad, b2_grad] = backpropagation(w1,b1,w2,b2,input_samples,output_samples)
  % 1 - Set the activation a0 for the input layer. 
  a0 = input_samples;
  
  % 2 - Feedforward
  [z1,z2,a1,a2] = feedforward(w1, b1, w2, b2, a0);
  
  % 3 - Output error
  delta_2 = sum(bsxfun(@times, a2-output_samples, sigmoid_gradient(z2))')';
  
  % 4 - Backpropagate the error
  delta_1 = (w2'*delta_2)';
  
  %5 - Set the gradient of the parameters
  
  size(delta_1)
  size(a1)
  
  w1_grad = a1.*delta_1;
  b1_grad = delta_1;
  w2_grad = a2.*delta_2;
  b2_grad = delta_2;
  
  
  123
  
end

function [z1,z2,a1,a2] = feedforward(w1, b1, w2, b2, a0)
  z1 = w1*a0 + b1;
  a1 = sigmoid(z1);
  z2 = w2*a1 + b2;
  a2 = sigmoid(z2);
end

function a = sigmoid(z)
  a = ones(size(z))./(ones(size(z))+exp(-z));
end

function grad = sigmoid_gradient(z)
  grad = sigmoid(z).*(ones(size(z))-sigmoid(z));
end

function [input_batch, output_batch] = CreateBatch(batch_size, input, output)
  indices = randperm(50000, batch_size);
  
  input_batch = input(:, indices);
  output_batch = output(:, indices);
end
