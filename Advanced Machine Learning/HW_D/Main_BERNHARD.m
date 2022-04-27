clear all
clc
X = load('training-set.csv'); %This could simply have been readmatrix in a newer version of Matlab.

%For X, the coordinates come in rows, and the time steps in columns.

n = 3; %Number of input neurons
N = 500; %Number of reservoir neurons
T = length(X); %Time steps to train
k = 0.01;

w_in = InitializeWIN(N,n); %Weights connecting input neurons to reservoir neurons.
w_res = InitializeW(N); %Reservoir weights, connects reservoir neurons between themselves.
%THE MATRICES ABOVE NEVER CHANGE DUE TO TRAINING. THEY ARE RANDOMLY
%INITIALIZED AND STAY THAT WAY. 

w_out = zeros(n,N); %Weights between reservoir neurons and output neurons.

%X(t) are input neuron states.
%r(t) are reservoir neuron states.
%O(t) are output neuron states.

r = zeros(N,1);
R = zeros(N,T); %Will have all the time steps.

%Training network
for t = 1:T
   r = tanh(w_res*r + w_in*X(:,t)); %Compute r for this time step
   R(:,t) = r;
end

%Ridge regression
I = eye(N);
w_out = X*transpose(R)*(R*transpose(R) + k.*I)^(-1);

%%

r = zeros(N,1); %Reset r.
X = load('test-set-6.csv');
T = length(X);

%Feed test data.
for t = 1:T
    r = tanh(w_res*r + w_in * X(:,t));
end

%Predict 
prediction = zeros(3,500);
O = X(:,T); %Start with the output being the last given point.
for t = 1:500
   r = tanh(w_res*r + w_in * O); %Compute r for this time step.
   O = w_out*r;
   prediction(:,t) = O;
end

predictionY = prediction(2,:);
%csvwrite('prediction.csv', predictionY);
plot3(prediction(1,:),prediction(2,:),prediction(3,:))

