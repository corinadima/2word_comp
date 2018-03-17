require 'paths' -- torch module for file path handling
dataLoader = require 'utils.dataLoader'

local compose_utils = {}

function compose_utils:loadDatasets(datasetDir, minNum)
	print('==> loading datasets...')
	
	local trainSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "train.txt"), " ")
	local devSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "dev.txt"), " ")
	local testSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "test.txt"), " ")
	local fullSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "full.txt"), " ")
	print('==> dataset loaded, train size:', trainSet:size(),
	  ' dev size', devSet:size(), ' test size', testSet:size(), ' full size', fullSet:size())


	return trainSet, devSet, testSet, fullSet
end	

function compose_utils:loadCMHDense(datasetDir, embeddingsId, size)
	print('==> loading dense embeddings of size ' .. size .. '...')
	local cmhEmbeddingsPath = paths.concat('data', datasetDir, 'embeddings', embeddingsId, embeddingsId .. '.' .. size .. 'd_cmh.dm')
	local cmhDictionary, cmhEmbeddings = dataLoader.loadDenseMatrix(cmhEmbeddingsPath)
	print('==> embeddings loaded, size:', cmhEmbeddings:size())

	print('==> norms')
	print(cmhEmbeddings[1])
	print(torch.norm(cmhEmbeddings[1]))
	print(torch.norm(cmhEmbeddings[2]))
	print(torch.norm(cmhEmbeddings[3]))

	return cmhDictionary, cmhEmbeddings
end

-- for Matrix, FullAdd
function compose_utils:createCMH2TensorDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local input = torch.zeros(2, sz)
		input[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		input[2] = cmhEmbeddings[tensorData[i][2]]:clone()


		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {input, output}
	end
	return dataset
end

-- for LexicalFunction
function compose_utils:createCMHM1V2Dataset(tensorData, cmhEmbeddings)
	local dataset = {}

	-- matrix index for the modifier, vector for the head

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local modifierIndex = torch.zeros(1)
		modifierIndex[1] = tensorData[i][1]

		local hSz = cmhEmbeddings[tensorData[i][2]]:size()

		local headVector = torch.Tensor(1, hSz[1], 1)
		headVector[1] = cmhEmbeddings[tensorData[i][2]]:clone()

		local dualInput = {modifierIndex, headVector} -- first matrix index - for modifier, then vector - for head

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

-- for FullLex
function compose_utils:createCMH_M2V1M1V2Dataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local vectors = torch.zeros(2, sz, 1) -- normal ordering for vectors, first the modifier, than the head
		vectors[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		vectors[2] = cmhEmbeddings[tensorData[i][2]]:clone()

		local matrixIndices = torch.zeros(2,1)

		matrixIndices[1] = tensorData[i][2] -- switch indices for matrices, we want to compute Ba; Ab
		matrixIndices[2] = tensorData[i][1]

		local dualInput = {matrixIndices, vectors} -- first matrix indices, then vectors

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

-- for Mask models
function compose_utils:createCMH_M1V1M2V2Dataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local vectors = torch.zeros(2, sz, 1) -- normal ordering for vectors, first the modifier, than the head
		vectors[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		vectors[2] = cmhEmbeddings[tensorData[i][2]]:clone()

		local matrixIndices = torch.zeros(2,1)

		matrixIndices[1] = tensorData[i][1]
		matrixIndices[2] = tensorData[i][2]

		local dualInput = {matrixIndices, vectors} -- first matrix indices, then vectors

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end


return compose_utils
