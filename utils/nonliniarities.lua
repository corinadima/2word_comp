require 'nn'

-- Wrapper for nonlinearities

nonliniarities = {}

function nonliniarities:tanhNonlinearity()
	return nn.Tanh()
end

function nonliniarities:sigmoidNonlinearity()
	return nn.Sigmoid()
end

function nonliniarities:reLUNonlinearity()
	return nn.ReLU()
end

function nonliniarities:identityNonlinearity()
	return nn.Identity()
end

return nonliniarities