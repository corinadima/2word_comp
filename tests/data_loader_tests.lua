require 'cutorch'
require 'cunn'
require 'cudnn'

local compose_utils = require 'utils.compose_utils'

local data_tests = torch.TestSuite()
local tester = torch.Tester()

function data_tests.l2_row_norm()
	local emb = torch.Tensor(3, 2):cuda()
	emb[1] = torch.Tensor({2,4})
	emb[2] = torch.Tensor({3,4})
	emb[3] = torch.Tensor({4,8})

	local expected = torch.Tensor(3, 2):cuda()
	expected[1] = torch.Tensor({0.4472, 0.8944})
	expected[2] = torch.Tensor({0.6, 0.8})
	expected[3] = torch.Tensor({0.4472, 0.8944})

	emb_renorm = compose_utils:normalizeEmbeddings(emb, 'l2_row')
	tester:assertTensorEq(emb_renorm, expected, 1e-2, 'incorrect l2 norms')

	norms = torch.norm(emb_renorm, 2, 2)
	tester:assertTensorEq(norms, torch.ones(3,1):cuda(), 1e-2, 'expected unit norms')
end

function data_tests.l2_col_norm()
	local emb = torch.Tensor(3, 2):cuda()
	emb[1] = torch.Tensor({2,4})
	emb[2] = torch.Tensor({3,4})
	emb[3] = torch.Tensor({4,8})

	local expected = torch.Tensor(3, 2):cuda()
	expected[1] = torch.Tensor({0.3713, 0.4082})
	expected[2] = torch.Tensor({0.5570 , 0.4082})
	expected[3] = torch.Tensor({0.7427, 0.8165})

	emb_renorm = compose_utils:normalizeEmbeddings(emb, 'l2_col')
	tester:assertTensorEq(emb_renorm, expected, 1e-2, 'incorrect l2_col norms')

	norms = torch.norm(emb_renorm, 2, 1)
	tester:assertTensorEq(norms, torch.ones(1, 2):cuda(), 1e-2, 'expected column unit norms')
end

function data_tests.none_norm()
	local emb = torch.Tensor(3, 2):cuda()
	emb[1] = torch.Tensor({2,4})
	emb[2] = torch.Tensor({3,4})
	emb[3] = torch.Tensor({4,8})

	local expected = emb

	emb_renorm = compose_utils:normalizeEmbeddings(emb, 'none')
	tester:assertTensorEq(emb_renorm, expected, 1e-2, 'expected an un-normalized matrix')
end

function data_tests.l1_row_norm()
	local emb = torch.Tensor(3, 2):cuda()
	emb[1] = torch.Tensor({-2,4})
	emb[2] = torch.Tensor({3,4})
	emb[3] = torch.Tensor({4,8})

	local expected = torch.Tensor(3, 2):cuda()
	expected[1] = torch.Tensor({-0.3333, 0.6666})
	expected[2] = torch.Tensor({0.4285, 0.5714})
	expected[3] = torch.Tensor({0.3333, 0.6666})

	emb_renorm = compose_utils:normalizeEmbeddings(emb, 'l1_row')
	tester:assertTensorEq(emb_renorm, expected, 1e-2, 'incorrect l1 row norms')
end

function data_tests.l1_col_norm()
	local emb = torch.Tensor(3, 2):cuda()
	emb[1] = torch.Tensor({-2,4})
	emb[2] = torch.Tensor({3,4})
	emb[3] = torch.Tensor({4,8})

	local expected = torch.Tensor(3, 2):cuda()
	expected[1] = torch.Tensor({-0.2222, 0.25})
	expected[2] = torch.Tensor({0.3333, 0.25})
	expected[3] = torch.Tensor({0.4444, 0.5})

	emb_renorm = compose_utils:normalizeEmbeddings(emb, 'l1_col')
	tester:assertTensorEq(emb_renorm, expected, 1e-2, 'incorrect l1 col norms')
end

tester:add(data_tests)
tester:run()
