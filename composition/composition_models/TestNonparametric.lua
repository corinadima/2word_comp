require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.HeadOnly'
require 'composition/composition_models.ModifierOnly'
require 'composition/composition_models.Addition'
require 'composition/composition_models.Multiplication'

require 'iterators.BatchIterator'

local composition_tests = torch.TestSuite()
local tester = torch.Tester()

local gpuid = 1
print('using CUDA on GPU ' .. gpuid .. '...')
cutorch.setDevice(gpuid)
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)


function composition_tests.Addition()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.Addition(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()


	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{7,9,11,13,15}
	expected_output[2] = torch.Tensor{9,11,13,15,17}
	tester:eq(output, expected_output, "output and expected_output should be equal")
end


function composition_tests.Head()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.HeadOnly(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()


	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{6,7,8,9,10}
	expected_output[2] = torch.Tensor{7,8,9,10,11}

	tester:eq(output, expected_output, "output and expected_output should be equal")
end

function composition_tests.Modifier()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.ModifierOnly(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()


	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{1,2,3,4,5}
	expected_output[2] = torch.Tensor{2,3,4,5,6}

	tester:eq(output, expected_output, "output and expected_output should be equal")
end

function composition_tests.Multiplication()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.Multiplication(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()


	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{6,14,24,36,50}
	expected_output[2] = torch.Tensor{14,24,36,50,66}

	tester:eq(output, expected_output, "output and expected_output should be equal")
end


tester:add(composition_tests)
tester:run()
