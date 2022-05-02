function w = InitializeW(N)
    w = zeros(N,N);
    for i = 1:N
       for j = 1:N
          w(i,j) = 2*(rand-0.5);
       end
    end
end