require 'nn'

-- Wrapper for nonlinearities

nonliniarities = {}

function nonliniarities:tanhNonlinearity()
	return nn.Tanh()
end

function nonliniarities:sigmoidNonlinearity()
	return nn.Sigmoid()
end

function nonliniarities:ReLUNonlinearity()
	return nn.ReLU()
end

function nonliniarities:ReLU6Nonlinearity()
	return nn.ReLU6()
end

function nonliniarities:identityNonlinearity()
	return nn.Identity()
end

function nonliniarities:PReLUNonlinearity()
	return nn.PReLU()
end

function nonliniarities:ELUNonlinearity()
	return nn.ELU()
end

return nonliniarities