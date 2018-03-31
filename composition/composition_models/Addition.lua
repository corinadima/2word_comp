local compose_utils = require 'utils.compose_utils'
--
-- Addition composition model
--
-- 

local Addition, parent = torch.class('torch.Addition', 'torch.CompositionModel')

function Addition:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "addition"
	self.isTrainable = false	
end

function Addition:architecture(config)
	parent:architecture(config)
	
	print("# Addition; vectors u and v as input; p = u + v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();

	local added = nn.CAddTable()({u, v})

	self.mlp = nn.gModule({u, v}, {added})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function Addition:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end