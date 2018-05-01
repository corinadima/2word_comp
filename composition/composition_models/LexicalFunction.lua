local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'

--
-- LexicalFunction composition model
--
-- 

local LexicalFunction, parent = torch.class('torch.LexicalFunction', 'torch.CompositionModel')

function LexicalFunction:__init(inputs, outputs, vocab_size)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.vocab_size = vocab_size
	self.name = "lexfunc"
	self.isTrainable = true
	self.iteratorType = 'table'
end

function LexicalFunction:architecture(config)
	parent:architecture(config)
	
	print("# LexicalFunction; vector v as input; modification matrix U is learned for each word individually; p = U * v ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local ltInit = torch.Tensor(self.vocab_size, self.inputs/2 * self.inputs/2)
	for i = 1, self.vocab_size do
		local lt = nn.Linear(self.inputs/2, self.inputs/2):init('weight', nninit.eye):init('weight', nninit.addNormal, 0, 1e-4)
		ltInit[i] = lt.weight
	end

	local u = nn.Identity()(); -- index
	local v = nn.Identity()(); -- vector

	local sel = nn.SelectTable(2)({u,v});

	local U = nn.LookupTable(self.vocab_size,self.inputs/2*self.inputs/2):init('weight', 
					nninit.copy, ltInit)(u):annotate{name='U'}
	self.U = nn.Dropout(config.dropout)(U);

	local reshape = nn.Reshape(self.inputs/2,self.inputs/2,true)({self.U})

	local mul = nn.MM()({reshape, sel})
	local reshape_mul = nn.Reshape(self.outputs,true)({mul})

	self.mlp = nn.gModule({u,v}, {reshape_mul})


	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	collectgarbage()

	return self.mlp
end

function LexicalFunction:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMHM1V2Dataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMHM1V2Dataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMHM1V2Dataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMHM1V2Dataset(fullSet, cmhEmbeddings)
end
