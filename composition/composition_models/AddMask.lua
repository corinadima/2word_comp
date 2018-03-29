local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'

--
-- Additive Mask composition model
--
-- 

local AddMask, parent = torch.class('torch.AddMask', 'torch.CompositionModel')

function AddMask:__init(inputs, outputs, vocab_size)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.vocab_size = vocab_size
	self.name = "AddMask"
	self.isTrainable = true
	self.iteratorType = 'quad'
end

function AddMask:architecture(config)
	print("# AddMask; vectors u,v as input; vector masks learned for each word individually; p = u' * u + v'' * v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local modifier_mask_idx = nn.Identity()();
	local u = nn.Identity()();
	local head_mask_idx = nn.Identity()();
	local v = nn.Identity()();

	local sel_u = nn.SelectTable(2)({modifier_mask_idx,u,head_mask_idx,v});
	local sel_v = nn.SelectTable(4)({modifier_mask_idx,u,head_mask_idx,v});

	local modifier_mask = nn.LookupTable(self.vocab_size,self.inputs/2):init('weight', nninit.constant, 1)(modifier_mask_idx):annotate{name="modifier_mask"}
	local head_mask = nn.LookupTable(self.vocab_size,self.inputs/2):init('weight', nninit.constant, 1)(head_mask_idx):annotate{name="head_mask"}

	local reshape_mm = nn.Reshape(self.inputs/2, 1, true)({modifier_mask})
	local reshape_hm = nn.Reshape(self.inputs/2, 1, true)({head_mask})

	local mul_1 = nn.CMulTable()({reshape_mm, sel_u})
	local mul_2 = nn.CMulTable()({reshape_hm, sel_v})

	local join = nn.CAddTable()({mul_1, mul_2})
	local reshape_join = nn.Reshape(self.outputs, true)({join})

	self.mlp = nn.gModule({modifier_mask_idx,u,head_mask_idx,v}, {reshape_join})


	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function AddMask:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH_M1V1M2V2Dataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH_M1V1M2V2Dataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH_M1V1M2V2Dataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH_M1V1M2V2Dataset(fullSet, cmhEmbeddings)
end
