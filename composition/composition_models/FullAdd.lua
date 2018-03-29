local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'
--
-- FullAdd composition model
--
-- 

local FullAdd, parent = torch.class('torch.FullAdd', 'torch.CompositionModel')

function FullAdd:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.isTrainable = true	
	self.name = "fulladd"
end

function FullAdd:architecture(config)
	print("# FullAdd; vector u and v are each multiplied via W1 and W2, two nxn matrices; no biases ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();

	local W1 = nn.Linear(self.inputs/2, self.outputs):init('weight', nninit.normal, 0, 1e-4):noBias()({u}):annotate{name="W1"}
	local W2 = nn.Linear(self.inputs/2, self.outputs):init('weight', nninit.normal, 0, 1e-4):noBias()({v}):annotate{name="W2"}

	self.W1 = W1
	self.W2 = W2
	
	local added = nn.CAddTable(2)({W1, W2})
	
	self.mlp = nn.gModule({u, v}, {added})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function FullAdd:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end
