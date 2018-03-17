require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.Dilation'
require 'composition/composition_models.WeightedAddition'

require 'iterators.BatchIterator'

local composition_tests = torch.TestSuite()
local tester = torch.Tester()

local gpuid = 1
print('using CUDA on GPU ' .. gpuid .. '...')
cutorch.setDevice(gpuid)
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)


function composition_tests.WeightedAddition()
	local sz = 5
	local config = {batchSize = 2}
	local composition_model = torch.WeightedAddition(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

	-- set lambda and beta to 1 => same as Addition
	parameters = torch.CudaTensor({2})
    for _, node in ipairs(mlp.forwardnodes) do
      	if node.data.annotations.name == 'lambda' or node.data.annotations.name == 'beta' then
        	node.data.module.weight:fill(1.0)
    	end
    end
    parameters, gradParameters = mlp:getParameters()
    print("parameters", parameters)

	local u = torch.Tensor({{1,2,3,4,5},{2,3,4,5,6}}):cuda()
	local v = torch.Tensor({{6,7,8,9,10}, {7,8,9,10,11}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{7,9,11,13,15}
	expected_output[2] = torch.Tensor{9,11,13,15,17}
	tester:eq(output, expected_output, "output and expected_output should be equal")
end


function composition_tests.Dilation()
	local sz = 3
	local config = {batchSize = 2}
	local composition_model = torch.Dilation(sz * 2, sz)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

	-- set lambda to 2	
	parameters = torch.CudaTensor({2})
    for _, node in ipairs(mlp.forwardnodes) do
      	if node.data.annotations.name == 'lambda' then
        	node.data.module.weight:fill(2.0)
    	end
    end
    parameters, gradParameters = mlp:getParameters()
    print("parameters", parameters)

	local u = torch.Tensor({{1,2,3},{2,3,4}}):cuda()
	local v = torch.Tensor({{6,7,8}, {7,8,9}}):cuda()
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,3):cuda()
	expected_output[1] = torch.Tensor{128,186,244}
	expected_output[2] = torch.Tensor{351,454,557}

	tester:eq(output, expected_output, "output and expected_output should be equal")
end


tester:add(composition_tests)
tester:run()
