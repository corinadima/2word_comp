local compose_utils = require 'utils.compose_utils'
--
-- ModifierOnly composition model
--
-- 

local ModifierOnly, parent = torch.class('torch.ModifierOnly', 'torch.CompositionModel')

function ModifierOnly:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "modifier"
	self.isTrainable = false	
end

function ModifierOnly:architecture(config)
	print("# ModifierOnly; vectors u and v as input; head vector (v) is discarded; p = u ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();

	local sel = nn.SelectTable(1)({u,v});
	
	self.mlp = nn.gModule({u, v}, {sel})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function ModifierOnly:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end