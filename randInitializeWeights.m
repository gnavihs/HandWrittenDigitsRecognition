function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   W is set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%

W = zeros(L_out, 1 + L_in);
epsilon = sqrt(6)/sqrt(L_in+L_in);

W = rand(L_out, 1 + L_in)*2*epsilon - epsilon;
% =========================================================================

end
