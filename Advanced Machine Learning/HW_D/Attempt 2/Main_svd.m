clear all
clc
predictionPerformance = Predict(1,0,1); %0.06
disp("The network predicted for " + num2str(predictionPerformance) + " time steps.")
%Noted about spectral radius but I expect the maximal singular value to
%reflect the same properties: A larger maximal singular value makes the
%network value long past history more compared to a smaller maximal
%singular value. In general we see that in one dimension it is more
%important to have a large enough maximal singular value, while in three
%dimensions it is not as stingy, and a lower value works better. 

%My observation is that predicting only one dimension is more difficult
%compared to predicting all dimensions, so for one dimension it may be more
%essential to "memorize" the pattern on a macroscopic scale rather than
%"learn" how the differential equations operate in all its dimensions.
%%
ifPlot = 0;
%Seems to work well at maxSing = 0.05.
nDimensions = 3;
maxSingVec = 1.5:0.01:2.7;
repeats = 2;
oneDimRes = zeros(repeats,length(maxSingVec));


for i = 1:length(maxSingVec)
    for repeat = 1:repeats
        maxSing = maxSingVec(i);
        disp(maxSing)
        oneDimRes(repeat, i) = Predict(nDimensions,maxSing,ifPlot);
    end
end


plot(maxSingVec, sum(oneDimRes)/repeats)
xlabel("Maximal singular value in w_{res}")
ylabel("Time steps predicted")
title(num2str(nDimensions)+ " dimensions")

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
        disp(max(svd(w_res)))
    else
        maxSingCurrent = max(svd(w_res));
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
            
            figure
            subplot(3,1,1)
            plot(O(1,:))
            hold on
            plot(testSet(1,:))
            legend("Start","Prediction","Actual");
            subplot(3,1,2)
            plot(O(2,:))
            hold on
            plot(testSet(2,:))
            legend("Start","Prediction","Actual");
            subplot(3,1,3)
            plot(O(3,:))
            hold on
            plot(testSet(3,:))
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