function w = InitializeW(N)
    w = zeros(N,N);
    for i = 1:N
       for j = 1:N
          w(i,j) = 2*rand-1;
       end
    end
end