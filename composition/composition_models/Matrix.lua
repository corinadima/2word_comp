local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'

--
-- Matrix composition model
--
-- 

local Matrix, parent = torch.class('torch.Matrix', 'torch.CompositionModel')

function Matrix:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.isTrainable = true	
	self.name = "matrix"
end

function Matrix:architecture(config)
	print("# Matrix; vector u and v are concatenated and composed through the global matrix W (size 2nxn); p = g(W[u;v] + b)")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()()
	local v = nn.Identity()()

	local join = nn.JoinTable(2)({u, v})

	local W = nn.Linear(self.inputs, self.outputs):init('weight', nninit.normal, 0, 1e-4)({join}):annotate{name="W"}
	local nonlinearity = self.config.nonlinearity({W})
	
	self.mlp = nn.gModule({u, v}, {nonlinearity})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function Matrix:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end
