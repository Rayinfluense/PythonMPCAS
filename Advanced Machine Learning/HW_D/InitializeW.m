function w = InitializeW(N)
    w = zeros(N,N);
    for i = 1:N
       for j = 1:N
          %w(i,j) = normrnd(0,0.004);
          w(i,j) = normrnd(0,0.1);
       end
    end
end