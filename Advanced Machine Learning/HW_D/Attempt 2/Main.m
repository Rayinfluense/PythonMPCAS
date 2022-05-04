clear all
clc
Predict(1,0.06,1)
%%
ifPlot = 0;
%Seems to work well at maxSing = 0.05.
nDimensions = 1;
maxSingVec = 0.03:0.001:0.1;
repeats = 15;
oneDimRes = zeros(repeats,length(maxSingVec));


parfor i = 1:length(maxSingVec)
    for repeat = 1:repeats
        maxSing = maxSingVec(i);
        disp(maxSing)
        oneDimRes(repeat, i) = Predict(nDimensions,maxSing,ifPlot);
    end
end


plot(maxSingVec, sum(oneDimRes)/repeats)
xlabel("Maximal singular value in w_{res}")
ylabel("Time steps predicted")
title(num2str(nDimensions), " dimensions")

function performance = Predict(nDimensions, maxSing, ifPlot)
    %nDimensions = 1; %1 or 3 only.
    XFull = Generate_Lorentz();
    if nDimensions == 1
        XFull = XFull(1,:);
    end
    setLength = size(XFull,2);
    trainRange = 1:floor(setLength*0.82);
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

    %Calculate spectral radius of matrix
    specRad = max(abs(eig(w_res))); %We want w_res to be divided by its spectral radius
    if maxSing == 0
        w_res = w_res./(specRad);
        disp(max(max(w_res)))
    else
        maxSingCurrent = max(max(w_res));
        w_res = w_res./(maxSingCurrent)*maxSing;
    end
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
    O = zeros(n, T);

    u = trainingSet(:,end);

    for t = 1:T
       x = (1-a)*x + a * tanh(w_in*[1;u] + w_res*x);
       u = w_out*[1;u;x];
       O(:,t) = u;
    end

    if ifPlot
        if nDimensions == 3
            plot3(testSet(1,1),testSet(2,1),testSet(3,1),'o','MarkerSize',10)
            hold on
            plot3(O(1,:),O(2,:),O(3,:))
            hold on
            plot3(testSet(1,:),testSet(2,:),testSet(3,:),'--','Color','black')
            legend("Start","Prediction","Actual");
        end

        if nDimensions == 1
            plot(O)
            hold on
            plot(testSet)
            legend("Prediction","Actual");
        end
    end
    performance = EvaluatePerformance(testSet,O);
end