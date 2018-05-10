local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'

--
-- MultiMatrix composition model
--
-- 

local MultiMatrix, parent = torch.class('torch.MultiMatrix', 'torch.CompositionModel')

function MultiMatrix:__init(inputs, outputs, no_matrices)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "MultiMatrix"
	self.isTrainable = true
	self.no_matrices = no_matrices
end

function MultiMatrix:architecture(config)
	parent:architecture(config)
	
	print("# MultiMatrix; vectors u,v as input; composition via no_matrices instances of the matrix model, followed by a nonlinearity and another matrix model;")
	print("# layer 1 matrices have size 2n x n; the large layer 2 matrix has size (no_matrices x n) x n")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();
	local join = nn.JoinTable(2)({u, v})

	local M = nn.ConcatTable()
	for i = 1, self.no_matrices do
		local lt = nn.Linear(self.inputs, self.outputs):init('weight', nninit.normal, 0, 1e-4)
		M:add(lt)
	end

	M = M({join})

	local join_M = nn.JoinTable(2)({M})
	local nonlinearity = self.config.nonlinearity(nn.Dropout(config.dropout)({join_M}))
	local W = nn.Linear(self.no_matrices * self.outputs, self.outputs):init('weight', nninit.normal, 0, 1e-4)({nonlinearity}):annotate{name="W"}

	self.mlp = nn.gModule({u, v}, {W})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function MultiMatrix:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end
