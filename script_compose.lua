require 'cutorch'
require 'cunn'
require 'cudnn'

require 'nngraph'
require 'optim'
require 'paths'
require 'xlua'

local sh = require 'sh' -- from https://github.com/zserge/luash

local lua_utils = require 'utils.lua_utils'
local compose_utils = require 'utils.compose_utils'
local nonliniarities = require 'utils.nonliniarities'

require 'composition/composition_models.CompositionModel'
require 'composition/composition_models.Matrix'
require 'composition/composition_models.FullAdd'
require 'composition/composition_models.HeadOnly'
require 'composition/composition_models.ModifierOnly'
require 'composition/composition_models.Addition'
require 'composition/composition_models.WeightedAddition'
require 'composition/composition_models.Multiplication'
require 'composition/composition_models.LexicalFunction'
require 'composition/composition_models.FullLex'
require 'composition/composition_models.Dilation'
require 'composition/composition_models.AddMask'
require 'composition/composition_models.WMask'

require 'iterators.BatchIterator'

print(_VERSION)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- command-line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('gsWordcomp: compositionality modelling')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'compositionality model to train: HeadOnly|ModifierOnly|Addition|WeightedAddition|Multiplication|Matrix|FullAdd|FullLex|AddMask|WMask')
cmd:option('-nonlinearity', 'tanh', 'nonlinearity to use, if needed by the architecture: identity|tanh|sigmoid|reLU')

cmd:option('-dim', 50, 'embeddings set, chosen via dimensionality: 50|100|200|300')
cmd:option('-dataset', 'german_compounds_nn_only_composition_dataset', 'dataset to use: english_compounds_composition_dataset|german_compounds_mixed_composition_dataset')
cmd:option('-mhSize', 7131, 'number of modifiers and heads in the dataset')
cmd:option('-embeddings', 'glove_decow14ax_all_min_100_vectors_raw', 'embeddings to use: glove_decow14ax_all_min_100_vectors_raw')
cmd:option('-normalization', 'none', 'normalization procedure to apply on the input embeddings: none|l2_row|l2_col|l1_row|l1_col')

-- cmd:option('-dim', 200, 'embeddings set, chosen via dimensionality: 300')
-- cmd:option('-dataset', 'german_compounds_mixed_composition_dataset', 'dataset to use: english_compounds_composition_dataset|german_compounds_mixed_composition_dataset')
-- cmd:option('-mhSize', 8580, 'number of modifiers and heads in the dataset: 8580|7131')
-- cmd:option('-embeddings', 'glove_decow14ax_all_min_100_vectors_l2_rows_cols', 'embeddings to use: glove_encow14ax_enwiki_8B.400k_l2norm_axis01|glove_decow14ax_all_min_100_vectors_l2norm_axis01|glove_decow14ax_all_min_100_vectors_raw')

-- cmd:option('-dim', 300, 'embeddings set, chosen via dimensionality: 50|100|200|300')
-- cmd:option('-dataset', 'english_compounds_composition_dataset', 'dataset to use: english_compounds_composition_dataset|german_compounds_mixed_composition_dataset')
-- cmd:option('-mhSize', 7646, 'number of modifiers and heads in the dataset: 7646|8580|7131')
-- cmd:option('-embeddings', 'glove_encow14ax_enwiki_9B.400k_l2_cols_rows', 'embeddings to use: ')
	
cmd:option('-gpuid', 1, 'GPU id or -1=use CPU')
cmd:option('-threads', 16, 'threads to use')
cmd:option('-criterion', 'mse', 'criterion to use: mse|cosine|abs')
cmd:option('-dropout', 0, 'dropout')
cmd:option('-extraEpochs', 5, 'extraEpochs for early stopping')
cmd:option('-batchSize', 100, 'mini-batch size (number between 1 and the size of the training data')
cmd:option('-outputDir', 'models', 'output directory to store the trained models')
cmd:option('-manual_seed', 1, 'manual seed for repeatable experiments')
cmd:option('-testDev', true, 'test model on dev dataset')
cmd:option('-testTest', false, 'test model on test dataset')
cmd:option('-testFull', false, 'test model on full dataset')
cmd:option('-lr', 0.01, 'learning rate')

cmd:text()

opt = cmd:parse(arg)
print(opt)

if (tonumber(opt.gpuid) > 0) then
	print('using CUDA on GPU ' .. opt.gpuid .. '...')
	cutorch.setDevice(opt.gpuid)
	torch.manualSeed(opt.manual_seed) 
	cutorch.manualSeed(opt.manual_seed, opt.gpuid)
	print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
else
	torch.setnumthreads(opt.threads)
	print('Running on CPU with ' .. opt.threads .. ' threads')
end
---------------------------------------------------------------------------
---------------------------------------------------------------------------

-- config
local config = {
	rundir = paths.concat(opt.outputDir, opt.dataset, opt.embeddings, opt.dim .. 'd'),
	batchSize = opt.batchSize,
	optimizer = 'adagrad',
	criterion = opt.criterion,
	normalization = opt.normalization,
	adagrad_config = {
		learningRate = opt.lr,
		learningRateDecay = 0,
		weightDecay = 0
	},
	sgd_config = {
		learningRate = 1e-2,
		learningRateDecay = 1e-4,
		weightDecay = 0.001,
		momentum = 0.9,
		nesterov = true,
	},
	earlyStopping = true,
	extraEpochs = opt.extraEpochs,
	manualSeed = opt.manual_seed,
	gpuid = tonumber(opt.gpuid),
	dropout = opt.dropout,
	cosineNeighbours = 0
}

local tf=os.date('%Y-%m-%d_%H-%M',os.time())

-- fix seed, for repeatable experiments
torch.manualSeed(config.manualSeed)

local configname = opt.model .. '_' .. opt.nonlinearity .. '_' .. config.optimizer ..
		 "_batch" .. config.batchSize .. "_" .. config.criterion .. "_" .. opt.normalization ..
		 "_lr_" .. tostring(opt.lr):gsub('%.', '-')

config.saveName = paths.concat(config.rundir, "model_" .. configname .. "_" .. tf)
xlua.log(config.saveName .. ".log")

print("==> config", config)
print("==> adagrad_config: ", config.adagrad_config)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- load data
local trainSet, devSet, testSet, fullSet = compose_utils:loadDatasets(opt.dataset, opt.minNum)
local cmhDictionary, cmhEmbeddings = compose_utils:loadCMHDense(opt.dataset, opt.embeddings, opt.dim, opt.normalization)

local sz = cmhEmbeddings:size()[2]
local vocab_size = opt.mhSize
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- composition models
local composition_models = {}
local nl = {}

composition_models['HeadOnly'] = torch.HeadOnly(sz * 2, sz)
composition_models['ModifierOnly'] = torch.ModifierOnly(sz * 2, sz)
composition_models['Addition'] = torch.Addition(sz * 2, sz)
composition_models['WeightedAddition'] = torch.WeightedAddition(sz * 2, sz)
composition_models['Multiplication'] = torch.Multiplication(sz * 2, sz)
composition_models['Matrix'] = torch.Matrix(sz * 2, sz)
composition_models['FullAdd'] = torch.FullAdd(sz * 2, sz)
composition_models['LexicalFunction'] = torch.LexicalFunction(sz * 2, sz, vocab_size)
composition_models['FullLex'] = torch.FullLex(sz * 2, sz, vocab_size)
composition_models['Dilation'] = torch.Dilation(sz * 2, sz)
composition_models['AddMask'] = torch.AddMask(sz * 2, sz, vocab_size)
composition_models['WMask'] = torch.WMask(sz * 2, sz, vocab_size)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- nonlinearities
nl['tanh'] = nonliniarities:tanhNonlinearity()
nl['sigmoid'] = nonliniarities:sigmoidNonlinearity()
nl['reLU'] = nonliniarities:reLUNonlinearity()
nl['identity'] = nonliniarities:identityNonlinearity()
---------------------------------------------------------------------------
---------------------------------------------------------------------------

local composition_model = composition_models[opt.model]
config.nonlinearity = nl[opt.nonlinearity]
local mlp = composition_model:architecture(config)
composition_model:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)

if (config.gpuid > 0) then
	mlp:cuda()
end

local timer = torch.Timer()
if (composition_model.isTrainable == true) then
	composition_model:train()
	print("==> Training ended");
else
	torch.save(config.saveName .. ".bin", mlp)
end

print("==> Saving predictions...")

composition_model:predict(opt.testDev, opt.testTest, opt.testFull, cmhDictionary, devSet, testSet, fullSet)

print('Time elapsed (real): ' .. lua_utils:secondsToClock(timer:time().real))
print('Time elapsed (user): ' .. lua_utils:secondsToClock(timer:time().user))
print('Time elapsed (sys): ' .. lua_utils:secondsToClock(timer:time().sys))

print("==> Model saved under " .. config.saveName .. ".bin");

function evaluate(subset, eval, config)
	print("running evaluation on " .. subset .. "...")
	local score = eval({
		composed=config.saveName .. '_' .. subset .. '.pred',
	 	dictionary=paths.concat('data', opt.dataset, 'embeddings', opt.embeddings, 
	 					opt.embeddings .. '.' .. opt.dim .. 'd_cmh.dm'),
		output=config.saveName .. '_' .. subset})
	print("Evaluation results:")
	print(score)
end

local eval = sh.command('python eva/composition_eval.py')

if (opt.testDev == true) then
	evaluate('dev', eval, config)
end

if (opt.testTest == true) then
	evaluate('test', eval, config)
end

if (opt.testFull == true) then
	evaluate('full', eval, config)
end
