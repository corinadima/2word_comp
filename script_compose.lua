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
require 'composition/composition_models.MultiMatrix'
require 'composition/composition_models.FullAdd'
require 'composition/composition_models.HeadOnly'
require 'composition/composition_models.ModifierOnly'
require 'composition/composition_models.Addition'
require 'composition/composition_models.WeightedAddition'
require 'composition/composition_models.Multiplication'
require 'composition/composition_models.LexicalFunction'
require 'composition/composition_models.FullLex'
require 'composition/composition_models.UncrossedFullLex'
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
cmd:text('2word_comp: compositionality modelling')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'compositionality model to train: HeadOnly|ModifierOnly|Addition|WeightedAddition|Multiplication|Matrix|FullAdd|FullLex|UncrossedFullLex|AddMask|WMask|MultiMatrix')
cmd:option('-nonlinearity', 'tanh', 'nonlinearity to use, if needed by the architecture: identity|tanh|sigmoid|ReLU|ReLU6|PReLU|ELU')

cmd:option('-dim', 50, 'embeddings set, chosen via dimensionality: 50|100|200|300')
cmd:option('-dataset', 'german_compounds_nn_only_composition_dataset', 'dataset to use: english_compounds_composition_dataset|german_compounds_mixed_composition_dataset')
cmd:option('-mhSize', 8580, 'number of modifiers and heads in the dataset 8580|7131')
cmd:option('-embeddings', 'glove_decow14ax_all_min_100_vectors_raw', 'embeddings to use: glove_decow14ax_all_min_100_vectors_raw')
cmd:option('-normalization', 'none', 'normalization procedure to apply on the input embeddings: none|l2_row|l2_col|l1_row|l1_col')
	
cmd:option('-gpuid', 1, 'GPU id or -1=use CPU')
cmd:option('-threads', 16, 'threads to use')
cmd:option('-criterion', 'cosine', 'criterion to use: mse|cosine|abs')
cmd:option('-dropout', 0, 'dropout')
cmd:option('-extraEpochs', 5, 'extraEpochs for early stopping')
cmd:option('-batchSize', 100, 'mini-batch size (number between 1 and the size of the training data')
cmd:option('-outputDir', 'models', 'output directory to store the trained models')
cmd:option('-manual_seed', 1, 'manual seed for repeatable experiments')
cmd:option('-testDev', true, 'test model on dev dataset')
cmd:option('-testTest', true, 'test model on test dataset')
cmd:option('-testFull', false, 'test model on full dataset')
cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-no_matrices', 120, 'number of distinct matrices to train in the MultiMatrix model')
cmd:option('-train_file', "train.txt", 'name of the train file to use')

cmd:text()

opt = cmd:parse(arg)

-- the lua devices are numbered starting from 1; however, when using CUDA_VISIBLE_DEVICES torch reports a single device with id 1
if (tonumber(opt.gpuid) >= 0) then
	print('using CUDA on GPU ' .. opt.gpuid .. '...')
	cutorch.setDevice(1)
	torch.manualSeed(opt.manual_seed) 
	cutorch.manualSeed(opt.manual_seed, 1)
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
	no_matrices = opt.no_matrices,
	lr = opt.lr,
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
	gpuid = tonumber(opt.gpuid) >= 0 and 1 or -1,
	dropout = opt.dropout,
	cosineNeighbours = 0,
	transformations = opt.no_matrices
}

local tf=os.date('%Y-%m-%d_%H-%M-%S',os.time())

-- fix seed, for repeatable experiments
torch.manualSeed(config.manualSeed)

local configname = opt.model .. '_' .. opt.nonlinearity .. '_' .. config.optimizer .. "_" .. opt.train_file ..
		 "_batch" .. config.batchSize .. "_" .. config.criterion .. "_" .. opt.normalization ..
		 "_lr_" .. tostring(opt.lr):gsub('%.', '-') .. "_dr_" .. tostring(opt.dropout):gsub('%.', '-') .. "_tr_" .. tostring(opt.no_matrices)

config.saveName = paths.concat(config.rundir, "model_" .. configname .. "_" .. tf)
xlua.log(config.saveName .. ".log")

print("==> opt", opt)
print("==> config", config)
print("==> adagrad_config: ", config.adagrad_config)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- load data
local trainSet, devSet, testSet, fullSet = compose_utils:loadDatasets(opt.dataset, opt.minNum, opt.train_file)
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
composition_models['UncrossedFullLex'] = torch.UncrossedFullLex(sz * 2, sz, vocab_size)
composition_models['Dilation'] = torch.Dilation(sz * 2, sz)
composition_models['AddMask'] = torch.AddMask(sz * 2, sz, vocab_size)
composition_models['WMask'] = torch.WMask(sz * 2, sz, vocab_size)
composition_models['MultiMatrix'] = torch.MultiMatrix(sz * 2, sz, opt.no_matrices)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- nonlinearities
nl['tanh'] = nonliniarities:tanhNonlinearity()
nl['sigmoid'] = nonliniarities:sigmoidNonlinearity()
nl['ReLU'] = nonliniarities:ReLUNonlinearity()
nl['ReLU6'] = nonliniarities:ReLU6Nonlinearity()
nl['identity'] = nonliniarities:identityNonlinearity()
nl['PReLU'] = nonliniarities:PReLUNonlinearity()
nl['ELU'] = nonliniarities:ELUNonlinearity()
---------------------------------------------------------------------------
---------------------------------------------------------------------------

local composition_model = composition_models[opt.model]
config.nonlinearity = nl[opt.nonlinearity]
local mlp = composition_model:architecture(config)
composition_model:data(trainSet, devSet, testSet, fullSet, cmhEmbeddings)

if (config.gpuid >= 0) then
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
