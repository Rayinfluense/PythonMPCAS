
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
