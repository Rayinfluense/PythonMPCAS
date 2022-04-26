function w_in = InitializeWIN(N,n)
    w_in = zeros(N,n);
    for i = 1:N
       for j = 1:n
          w_in(i,j) = 0.1*(2*rand-1);
       end
    end
end