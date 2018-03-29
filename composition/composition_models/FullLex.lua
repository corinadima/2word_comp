local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'

--
-- Full Lexical composition model
--
-- 

local FullLex, parent = torch.class('torch.FullLex', 'torch.CompositionModel')

function FullLex:__init(inputs, outputs, vocab_size)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.vocab_size = vocab_size
	self.name = "fulllex"
	self.isTrainable = true
	self.iteratorType = 'quad'
end

function FullLex:architecture(config)
	print("# FullLex; vectors u,v as input; matrices A_w learned for each word individually; p = g(W[A_v_idx * u; A_u_idx * v] + b) ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local ltInit = torch.Tensor(self.vocab_size, self.inputs/2 * self.inputs/2)
	for i = 1, self.vocab_size do
		local lt = nn.Linear(self.inputs/2, self.inputs/2):init('weight', nninit.eye):init('weight', nninit.addNormal, 0, 1e-4)
		ltInit[i] = lt.weight
	end

	local Av_idx = nn.Identity()();
	local u = nn.Identity()();
	local Au_idx = nn.Identity()();
	local v = nn.Identity()();

	local r_Av_idx = nn.Reshape(1,true)({Av_idx})
	local r_Au_idx = nn.Reshape(1,true)({Au_idx})

	local join = nn.JoinTable(2)({r_Av_idx, r_Au_idx})

	local Av_Au = nn.LookupTable(self.vocab_size,self.inputs/2*self.inputs/2):init('weight', nninit.copy, ltInit)({join})

	local splits = nn.SplitTable(2)({Av_Au})

	local reshape_Av = nn.Reshape(self.inputs/2,self.inputs/2,true)({nn.SelectTable(1){splits}})
	local reshape_Au = nn.Reshape(self.inputs/2,self.inputs/2,true)({nn.SelectTable(2){splits}})

	local mul_1 = nn.MM()({reshape_Av, u})
	local mul_2 = nn.MM()({reshape_Au, v})

	local join_mul = nn.JoinTable(2)({mul_1, mul_2})
	local reshape_join = nn.Reshape(self.inputs,true)({join_mul})

	local W = nn.Linear(self.inputs, self.outputs):init('weight', nninit.normal, 0, 1e-4)({reshape_join}):annotate{name="W"}
	local nonlinearity = nn.Tanh()({W})

	self.mlp = nn.gModule({Av_idx,u,Au_idx,v}, {nonlinearity})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function FullLex:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH_M2V1M1V2Dataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH_M2V1M1V2Dataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH_M2V1M1V2Dataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH_M2V1M1V2Dataset(fullSet, cmhEmbeddings)
end
