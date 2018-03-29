require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
local nninit = require 'nninit'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.Matrix'
require 'composition/composition_models.FullAdd'

require 'iterators.BatchIterator'

local composition_tests = torch.TestSuite()
local tester = torch.Tester()

local gpuid = 1
print('using CUDA on GPU ' .. gpuid .. '...')
cutorch.setDevice(gpuid)
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)


function composition_tests.Matrix()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.Matrix(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

    for _, node in ipairs(mlp.forwardnodes) do
      	if node.data.annotations.name == 'W' then
        	node.data.module.weight:fill(1.0)
    	end
    end

	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{1,1,1,1,1}
	expected_output[2] = torch.Tensor{1,1,1,1,1}
	tester:assertTensorEq(output, expected_output, 1e-2, "output and expected_output should be equal")
end

function composition_tests.FullAdd()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.FullAdd(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

    for _, node in ipairs(mlp.forwardnodes) do
      	if node.data.annotations.name == 'W1' or node.data.annotations.name == 'W2' then
        	node.data.module:init('weight', nninit.eye)
    	end
    end

	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{7,9,11,13,15}
	expected_output[2] = torch.Tensor{9,11,13,15,17}
	tester:assertTensorEq(output, expected_output, 1e-2, "output and expected_output should be equal")
end

tester:add(composition_tests)
tester:run()
