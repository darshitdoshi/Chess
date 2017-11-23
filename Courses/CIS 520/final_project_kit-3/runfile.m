load('train.mat')
load('validation.mat')
load('vocabulary.mat')

A = [X_train_bag, Y_train];
n = size(X_train_bag, 1);
A = A(randperm(n),:);

X_train = A(1:17092,1:size(X_train_bag,2));
label_train = A(1:17092,10001);
X_validate = A(17093:18092,1:size(X_train_bag,2));
label_validate = A(17093:18092,10001);

model = liblinear_train(label_train, X_train, '-s 0');
[predicted_labels, accuracy, prob_estimates] = predict(label_validate, X_validate, model, '-b 1');
err = performance_measure(predicted_labels, label_validate);

%TF-IDF
% X_train_tfidf = feature_selection(X_train);
% X_train = X_train_tfidf;

%Norm
X_train_norm = double(X_train>0);
% X_train_norm = bsxfun(@rdivide, X_train_norm, sum(X_train_norm,2));
model_norm = liblinear_train(label_train, X_train_norm, '-s 6 -c 0.7');
% X_validate_norm = double(X_validate>0); 
[predicted_labels_norm, accuracy_norm, prob_estimates_norm] = predict(label_validate, X_validate, model_norm, '-b 1');
err_norm = performance_measure(predicted_labels_norm, label_validate);