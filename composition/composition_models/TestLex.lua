require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.LexicalFunction'
require 'composition/composition_models.FullLex'

require 'iterators.BatchIterator'

local composition_tests = torch.TestSuite()
local tester = torch.Tester()

local gpuid = 1
print('using CUDA on GPU ' .. gpuid .. '...')
cutorch.setDevice(gpuid)
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)


function composition_tests.LexicalFunction()
	local sz = 5
	local config = {batchSize = 2}
	local vocabSize = 2
	local composition_model = torch.LexicalFunction(sz * 2, sz, vocabSize)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

	local u = torch.Tensor(2):cuda()
	u[1] = torch.Tensor({1})
	u[2] = torch.Tensor({2})
	local v = torch.Tensor(2,5,1):cuda()
	v[1] = torch.Tensor({6,7,8,9,10})
	v[2] = torch.Tensor({7,8,9,10,11})
	local input = {u, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{6,7,8,9,10}
	expected_output[2] = torch.Tensor{7,8,9,10,11}
	tester:assertTensorEq(output, expected_output, 1e-2, "output and expected_output should be equal")
end

function composition_tests.FullLex()
	local sz = 5
	local config = {batchSize = 2, nonlinearity = nn.Identity()}
	local vocabSize = 2
	local composition_model = torch.FullLex(sz * 2, sz, vocabSize)
	local mlp = composition_model:architecture(config)
	mlp:cuda()

    for _, node in ipairs(mlp.forwardnodes) do
      	if node.data.annotations.name == 'W' then
        	two_eyes = torch.cat(torch.eye(sz), torch.eye(sz), 2):cuda()
        	node.data.module.weight = two_eyes
    	end
    end

	local u_idx = torch.Tensor(2):cuda()
	u_idx[1] = torch.Tensor({1})
	u_idx[2] = torch.Tensor({2})
	local u = torch.Tensor(2,5,1):cuda()
	u[1] = torch.Tensor({1,2,3,4,5})
	u[2] = torch.Tensor({3,4,5,6,7})

	local v_idx = torch.Tensor(2):cuda()
	v_idx[1] = torch.Tensor({2})
	v_idx[2] = torch.Tensor({1})
	local v = torch.Tensor(2,5,1):cuda()
	v[1] = torch.Tensor({6,7,8,9,10})
	v[2] = torch.Tensor({7,8,9,10,11})
	local input = {u_idx, u, v_idx, v}
	local output = mlp:forward(input)

	local expected_output = torch.Tensor(2,5):cuda()
	expected_output[1] = torch.Tensor{7,9,11,13,15}
	expected_output[2] = torch.Tensor{10,12,14,16,18}
	tester:assertTensorEq(output, expected_output, 5e-1, "output and expected_output should be equal")
end



tester:add(composition_tests)
tester:run()
