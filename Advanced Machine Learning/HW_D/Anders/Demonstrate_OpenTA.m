
train = load('training-set.csv');
plot(train(2,:))
%%
test = load('test-set-9.csv');
plot(test(2,:))
%%

prediction = load('prediction_correct.csv');
prediction = transpose(prediction);


merge = [train(2,:),test(2,:),prediction];
t = 1:length(merge);
plot(t(1:length(train)),merge(1:length(train)))
hold on
plot(t(length(train)+1:length(train)+length(test)),merge(length(train)+1:length(train)+length(test)))
plot(t(length(train)+length(test)+1:end),merge(length(train)+length(test)+1:end))

%plot([train(2,:),test(2,:),prediction])


axis([1.9*10^4,2.1*10^4,-30,30])

legend('Training set','Test set','Prediction')