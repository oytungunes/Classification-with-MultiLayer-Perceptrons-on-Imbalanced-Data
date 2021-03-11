function [ normalized_matrix ] = normalize_data( input_matrix )


A=input_matrix;
centralized=A-ones(size(A,1),1)*mean(A);
normalized_matrix=centralized./(ones(size(centralized,1),1)*(std(centralized)));


end

