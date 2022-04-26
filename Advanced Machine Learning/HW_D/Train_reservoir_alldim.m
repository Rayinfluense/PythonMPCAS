clear all
clc

XFull = Generate_Lorentz();
%XFull = XFull(2,:);
trainEnd = 0.6;
testEnd = 0.9;

n = size(XFull,1); %Number of input neurons
N = 500; %Number of reservoir neurons
X = XFull(:,1:floor(length(XFull)*trainEnd));
T = length(X); %Time steps to train

k = 0.01;

w_in = InitializeWIN(N,n); %Weights connecting input neurons to reservoir neurons.
w_res = InitializeW(N); %Reservoir weights, connects reservoir neurons between themselves.
%THE MATRICES ABOVE NEVER CHANGE DUE TO TRAINING. THEY ARE RANDOMLY
%INITIALIZED AND STAY THAT WAY. 

w_out = zeros(n,N); %Weights between reservoir neurons and output neurons.

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

X = XFull(:,T+1:floor(length(XFull)*testEnd));
T = length(X);

%Feed test data.
for t = 1:T
    r = tanh(w_res*r + w_in * X(:,t));
end

%Predict 
prediction = zeros(size(XFull,1),length(XFull(2,floor(length(XFull)*testEnd):end)));
O = X(:,T); %Start with the output being the last given point.
prediction(:,1) = O;
for t = 2:length(prediction)
   r = tanh(w_res*r + w_in * O); %Compute r for this time step.
   O = w_out*r;
   prediction(:,t) = O;
end

prediction = prediction(2,:);
csvwrite('prediction.csv', prediction);

plot(prediction)
hold on

plot(XFull(2,floor(length(XFull)*testEnd):end))