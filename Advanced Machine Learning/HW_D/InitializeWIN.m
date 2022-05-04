function w_in = InitializeWIN(N,n)
    w_in = zeros(N,n);
    for i = 1:N
       for j = 1:n
          %w_in(i,j) = normrnd(0,0.002);
          w_in(i,j) = normrnd(0,0.02);
       end
    end
end