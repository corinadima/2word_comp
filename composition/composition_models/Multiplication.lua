local compose_utils = require 'utils.compose_utils'
--
-- Multiplication composition model
--
-- 

local Multiplication, parent = torch.class('torch.Multiplication', 'torch.CompositionModel')

function Multiplication:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "mul"
	self.isTrainable = false	
end

function Multiplication:architecture(config)
	print("# Multiplication; vectors u and v as input, multiplied component-wise; p = u * v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();

	local mul = nn.CMulTable()({u, v})

	self.mlp = nn.gModule({u, v}, {mul})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function Multiplication:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end