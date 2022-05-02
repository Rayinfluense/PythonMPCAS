function a = Generate_Lorentz()
    sigma = 10;
    r = 28;
    b = 8/3;
    dt = 0.01;
    endTime = 6000;

    x1 = 0.1;
    x2 = 0.2;
    x3 = 0.3;


    for t = 1:endTime
        x1(end+1) = x1(end) + (-sigma*x1(end) + sigma*x2(end))*dt;
        x2(end+1) = x2(end) + (-x1(end)*x3(end) +  r*x1(end) - x2(end))*dt;
        x3(end+1) = x3(end) + (x1(end)*x2(end) - b*x3(end))*dt;
    end

    %plot3(x1,x2,x3)
    a = [x1;x2;x3];
end