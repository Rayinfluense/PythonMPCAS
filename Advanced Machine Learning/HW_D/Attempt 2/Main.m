clear all
clc

dumbInTheHead = 0;
nDimensions = 1;
XFull = Generate_Lorentz();
if nDimensions == 1
    XFull = XFull(1,:);
end
setLength = size(XFull,2);
trainRange = 1:floor(setLength*0.6);
testRange = trainRange(end):setLength;

trainingSet = XFull(:,trainRange);
testSet = XFull(:,testRange);

n = nDimensions; %Number of input neurons
N = 800; %Number of reservoir neurons
T = length(trainingSet); %Time steps to train
k = 0.00001; %Parameter for ridge regression
a = 0.3;

w_in = InitializeWIN(N,n+1); %Weights connecting input neurons to reservoir neurons.
w_res = InitializeW(N); %Reservoir weights, connects reservoir neurons between themselves.

if dumbInTheHead
    w_in = zeros(N, 1 + n);
    increment = 11;
    add = 0;
    for i = 1:N
        for j = 1:(1 + n)
            w_in(i,j) = ((add/100)-0.5)*0.2;
            add = add + increment;
            increment = increment + 1;
            if (add >= 100)
                add = add - 100;
            end
            if(increment >= 50)
                increment = increment -47;
            end
        end
    end



    w_res = zeros(N,N);
    increment = 17;
    add = 0;
    for i = 1:N
        for j = 1:N
            w_res(i,j) = ((add/100)-0.5)*2;
            add = add + increment;
            increment = increment + 1;
            if (add >= 100)
                add = add - 100;
            end
            if(increment >= 50)
                increment = increment -47;
            end
        end
    end
end


%Calculate spectral radius of matrix
specRad = max(abs(eig(w_res))); %We want w_res to be divided by its spectral radius
w_res = w_res./specRad;

x = zeros(N,1);
X = zeros(1 + n + N,T); %Will have all the time steps.

%Training network
for t = 1:T   
   x = (1 - a)* x + a * tanh(w_res*x + w_in*[1;trainingSet(:,t)]); %Compute r for this time step
   X(:,t) = [1; trainingSet(:,t); x];
end

%Ridge regression
I = eye(1 + n + N);
w_out = [trainingSet(:,2:end),testSet(:,1)]*transpose(X)*(X*transpose(X) + k.*I)^(-1);

T = length(testSet);
Y = zeros(n, T);

u = trainingSet(end);

for t = 1:T
   x = (1-a)*x + a * tanh(w_in*[1;u] + w_res*x);
   u = w_out*[1;u;x];
   Y(:,t) = u;
end

if nDimensions == 3
    plot3(Y(1,:),Y(2,:),Y(3,:))
    hold on
    plot3(testSet(1,:),testSet(2,:),testSet(3,:))
end

if nDimensions == 1
    plot(Y)
    hold on
    plot(testSet)
end

