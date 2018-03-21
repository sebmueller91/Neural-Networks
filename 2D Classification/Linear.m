function Linear
  % Parameters
  n_samples = 500;
  
  % create data
  x = rand(2,n_samples);
  y = zeros(size(x));
  
  labels = (x(1,:) - x(2,:) > 0) + 1;
  for i=1:sum(x,2)
    y(labels(i),i) = 1;  
  end
  
  % Divide into training and test data
  x_train = x(:,1:n_samples/2);
  y_train = y(:,1:n_samples/2);
  labels_train = labels(1:n_samples/2);
  
  x_test = x(:,n_samples/2+1:n_samples);
  y_test = y(:,n_samples/2+1:n_samples);
  labels_test = labels(n_samples/2+1:n_samples);
  
  % Declare parameters and variables
  s0 = 2; % Size of the input layer
  s1 = 5; % Size of the hidden layer
  s2 = 2; % Size of the output layer
  
  % Train the network
  [w1,b1,w2,b2,train_percentages,test_percentages] = ...
    TrainTwoLayerNN(s0,s1,s2, x_train, y_train, labels_train, x_test, labels_test, 1);
  
  % Do final prediction on the test set
  final_percentage = CalcPredictionError(w1,b1,w2,b2,x_test,labels_test)*100
  
  % Plot data
  subplot(1,2,1) 
  scatter(x(1,(labels == 1)), x(2,(labels == 1)));
  hold on;
  scatter(x(1,(labels == 2)), x(2,(labels == 2)));
  
  subplot(1,2,2)
  x_vals = [1:length(train_percentages)];    
  plot(x_vals, train_percentages.*100);
  hold on;
  plot(x_vals, test_percentages.*100);
  legend('Train data calssification percentage', 'Test data calssification percentage','Location','southeast');
end

function percentage = CalcPredictionError(w1,b1,w2,b2,x,labels)
  [z1,z2,a1,a2] = feedforward(w1, b1, w2, b2, x);
  [~, predicted_labels] = max(a2);
  percentage = sum(labels == predicted_labels) / size(labels,2);
end

function [w1,b1,w2,b2,train_percentages,test_percentages] = ...
    TrainTwoLayerNN(s0,s1,s2, train_input, train_output, labels_train, test_input, labels_test, trace_error)
    
  % variables and parameters
  grad_norm = inf;
  %eta = 2;
  batch_size = 100;
  n_iterations = 1000;
  it = 0;
  train_percentages = zeros(1,n_iterations);
  test_percentages = zeros(1,n_iterations); 
  
  eta_half_life = 200;
  eta_0 = 10;
  lambda = 1;
  
  % initialize weights and biases randomly
  w1 = randn([s1,s0])./sqrt(s0); 
  b1 = randn(s1,1)./sqrt(s0);
  w2 = randn([s2,s1])./sqrt(s1);
  b2 = randn(s2,1)./sqrt(s1);
  
  % Stochastic Gradient Descent
  while (it < n_iterations)
    it = it + 1;
    
    % Calculate step width
    eta = eta_0*exp(-(log(2)/eta_half_life)*it);
    
    % Create batches 
    [input_batch, output_batch] = CreateBatch(batch_size, train_input, train_output);
    
    % Calculate derivative with backpropagation
    [w1_grad, b1_grad, w2_grad, b2_grad] = backpropagation(w1,b1,w2,b2,input_batch,output_batch, lambda);
    
    % Do one step in the direction of the derivative
    w1 = w1 - eta.*w1_grad;
    %w1 = (1-(eta*lambda)/batch_size).*w1 + eta.*w1_grad;
    b1 = b1 - eta.*b1_grad;
    w2 = w2 - eta.*w2_grad;
    %w2 = (1-(eta*lambda)/batch_size).*w2 + eta.*w2_grad;
    b2 = b2 - eta.*b2_grad;
    
    % Calculate the current error
    if (trace_error == 1)
      train_percentages(it) = CalcPredictionError(w1,b1,w2,b2,train_input,labels_train);
      test_percentages(it) = CalcPredictionError(w1,b1,w2,b2,test_input,labels_test);
    end
  end
  
  % Plot the error development 
  if (trace_error == 1)
    x = [1:n_iterations];    
    plot(x, train_percentages.*100);
    hold on;
    plot(x, test_percentages.*100);
    legend('Train data calssification percentage', 'Test data calssification percentage','Location','northwest');
  end    
end

function [w1_grad, b1_grad, w2_grad, b2_grad] = backpropagation(w1,b1,w2,b2,input_samples,output_samples, lambda)
  batch_size = size(input_samples,2);
  
  w1_grad = zeros(size(w1));
  b1_grad = zeros(size(b1));
  w2_grad = zeros(size(w2));
  b2_grad = zeros(size(b2));
  
  % 1 - Feedforward
  [z1_full,z2_full,a1_full,a2_full] = feedforward(w1, b1, w2, b2, input_samples);
  
  for i=1:batch_size
    % Fetch values of the current input sample
    x = input_samples(:,i);
    y = output_samples(:,i);  
    z1 = z1_full(:,i);
    z2 = z2_full(:,i);
    a1 = a1_full(:,i);
    a2 = a2_full(:,i);
    
    % 2 - Calculate output error
    %delta_2 = (a2-y).*activation_gradient(z2);
    delta_2 = x.*(activation(z2)-y);
    
    % 3 - Backpropagate the error
    delta_1 = (w2' * delta_2) .* activation_gradient(z1);
    
    % 4 - Update gradient
    %w1_grad = w1_grad + delta_1*x' + lambda * w1;
    w1_grad = w1_grad + delta_1*x';
    b1_grad = delta_1;
    %w2_grad = w2_grad + delta_2*a1' + lambda * w2;
    w2_grad = w2_grad + delta_2*a1';
    b2_grad = delta_2;
    
  end

  % Normalize the gradient
  w1_grad = w1_grad ./ batch_size ;
  b1_grad = b1_grad ./ batch_size;
  w2_grad = w2_grad ./ batch_size;
  b2_grad = b2_grad ./ batch_size;
  
  
end

function [z1,z2,a1,a2] = feedforward(w1, b1, w2, b2, a0)
  z1 = w1*a0 + b1;
  a1 = activation(z1);
  z2 = w2*a1 + b2;
  a2 = activation(z2);
end

function a = activation(z)
  a = ones(size(z))./(ones(size(z))+exp(-z));
  %a = tanh(z);
end

function grad = activation_gradient(z)
  grad = activation(z).*(ones(size(z))-activation(z));
  %grad = ones(size(z)) - tanh(z).^2;
end

function [input_batch, output_batch] = CreateBatch(batch_size, input, output)
  indices = randperm(size(input,2), batch_size);
  
  input_batch = input(:, indices);
  output_batch = output(:, indices);
end
