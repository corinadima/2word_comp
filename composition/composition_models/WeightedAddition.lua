local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'
--
-- WeightedAddition composition model
--
-- 

local WeightedAddition, parent = torch.class('torch.WeightedAddition', 'torch.CompositionModel')

function WeightedAddition:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "w_addition"
	self.isTrainable = true	
	self.printParameters = true
end

function WeightedAddition:architecture(config)
	parent:architecture(config)
	
	print("# WeightedAddition; vectors u and v as input; p = lambda*u + beta*v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();

	local lambda_u = nn.Mul():init('weight', nninit.normal, 0, 1e-4)({u}):annotate{name="lambda"}
	local beta_v = nn.Mul():init('weight', nninit.normal, 0, 1e-4)({v}):annotate{name="beta"}

	local added = nn.CAddTable()({lambda_u, beta_v})

	self.mlp = nn.gModule({u, v}, {added})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function WeightedAddition:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end