local compose_utils = require 'utils.compose_utils'
--
-- HeadOnly composition model
--
-- 

local HeadOnly, parent = torch.class('torch.HeadOnly', 'torch.CompositionModel')

function HeadOnly:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "head"
	self.isTrainable = false	
end

function HeadOnly:architecture(config)
	parent:architecture(config)
	
	print("# HeadOnly; vectors u and v as input; modifier vector (u) is discarded; p = v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local c1 = nn.Identity()();
	local c2 = nn.Identity()();

	local sel = nn.SelectTable(2)({c1,c2});
	
	self.mlp = nn.gModule({c1, c2}, {sel})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function HeadOnly:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end