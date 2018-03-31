local compose_utils = require 'utils.compose_utils'
local nninit = require 'nninit'
--
-- Dilation composition model
--
-- 

local Dilation, parent = torch.class('torch.Dilation', 'torch.CompositionModel')

function Dilation:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.name = "dilation"
	self.isTrainable = true	
	self.printParameters = true
end

function Dilation:architecture(config)
	parent:architecture(config)
	
	print("# Dilation; vectors u and v as input; p = (u dot u)v + (lambda - 1)(u dot v)u ")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	self.config = config

	local u = nn.Identity()();
	local v = nn.Identity()();
	local u_2 = nn.Identity()({u});-- quirk of nngraph node - cannot pass the same node as input twice to the same module

	local res_v = nn.Reshape(1,self.inputs/2,true)({v});
	local res_u = nn.Reshape(1,self.inputs/2,true)({u});

	local uu = nn.DotProduct()({u, u_2})
	local res_uu = nn.Reshape(1,1,true)({uu})
	local mul_1 = nn.MM()({res_uu,res_v})

	local uv = nn.DotProduct()({u, v})
	local res_uv = nn.Reshape(1,1,true)({uv})
	local mul_2 = nn.MM()({res_uv,res_u})

	local lambda = nn.Mul():init('weight', nninit.normal, 0, 1e-4)({mul_2}):annotate{name="lambda"}
	local lambda_minus_one = nn.CSubTable()({lambda, mul_2})

	local added = nn.CAddTable()({mul_1, lambda_minus_one})
	local reshape_added = nn.Reshape(self.outputs,true)({added})

	self.mlp = nn.gModule({u,v}, {reshape_added})

	print("==> Network configuration")
	print(self.mlp)
	print("==> Parameters size")
	print(self.mlp:getParameters():size())

	return self.mlp
end

function Dilation:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)
	self.trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	self.testDataset = compose_utils:createCMH2TensorDataset(testSet, cmhEmbeddings)
	self.devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)
	self.fullDataset = compose_utils:createCMH2TensorDataset(fullSet, cmhEmbeddings)
end