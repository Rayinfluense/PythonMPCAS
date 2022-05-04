function correctTimeSteps = EvaluatePerformance(sequence1, sequence2)
    %Allow a certain euclidian distance and the performance is how
    %many time steps the prediction stays within that distance.
    
    tolerance = 5^2;
    correctTimeSteps = 0;
    t = 1;
    errorSq = (sequence1(:,t) - sequence2(:,t)).^2;
    while errorSq < tolerance
        t = t + 1;
        errorSq = (sequence1(:,t) - sequence2(:,t)).^2;
        correctTimeSteps = correctTimeSteps + 1;
        if t == size(sequence1,2)
            break
        end
    end
end