require 'torch'
require 'gnuplot'

-- To debug an nngraph
-- nngraph.setDebug(true)
-- nngraph.annotateNodes() - add this just before creating the mlp (call to nn.gModule)


local CompositionModel = torch.class('torch.CompositionModel')

function CompositionModel:__init()
	self.isTrainable = true
	self.printParameters = false
	self.iteratorType = 'tensor'
	self.config = {}
	self.withCUDA = false

	batchIterator = require 'iterators.TensorBatchIterator'
end

function CompositionModel:architecture(config)
	if (config.criterion == 'mse') then
		self.criterion = nn.MSECriterion(false)
	elseif (config.criterion == 'abs') then
		self.criterion = nn.AbsCriterion()
		self.criterion.sizeAverage = false
	elseif (config.criterion == 'cosine') then
		self.criterion = nn.CosineEmbeddingCriterion()
		self.criterion.sizeAverage = false
	else
		error("Unknown criterion")
	end
end

function CompositionModel:data()
end

function CompositionModel:visualize(epoch)
end

function CompositionModel:doTest(module, criterion, config, dataset)
	module:evaluate()

	local testError = 0

	-- one pass 

	-- Create a batched iterator
	local testIter = batchIterator:initialize({
		data = dataset,
		randomize = false,
		cuda = self.withCUDA,
		batchSize = config.batchSize,
		inputShape = {2, self.inputs/2}
	})

	local nextBatch = testIter:nextBatch()

	while (nextBatch ~= nil) do

		-- test samples
		local predictions = module:forward(nextBatch.inputs)

		-- compute error

		local pred, tar

		if (config.criterion == 'cosine') then
			pred = {predictions, nextBatch.targets}
			if (self.withCUDA == true) then
				tar = torch.Tensor(nextBatch.currentBatchSize):fill(1):cuda()
			else
				tar = torch.Tensor(nextBatch.currentBatchSize):fill(1)
			end
		elseif (config.criterion == 'mse' or config.criterion == 'abs') then
			pred = predictions
			tar= nextBatch.targets
		else
			error("Unknown criterion")
		end

		-- print("prediction size ".. pred:size()[1] .. " " .. pred:size()[2])
		-- for i = 1,pred:size()[2] do
		-- 		print(string.format("%.4f %.4f", pred[1][i], tar[1][i]))
		-- end

		local err = self.criterion:forward(pred, tar)
		testError = testError + err

		nextBatch = testIter:nextBatch()
	end

	-- average error over the dataset
	testError = testError/dataset:size()

	return testError
end

function CompositionModel:train()
	print("iterator type " .. self.iteratorType)
	if (self.iteratorType == 'table') then
		batchIterator = require 'iterators.TableBatchIterator'
	elseif (self.iteratorType == 'quad') then
		batchIterator = require 'iterators.QuadBatchIterator'
	end

	print("Hyperparameter config")
	print(self.config)

	local config = self.config

	if (self.config.gpuid >= 0) then
		self.criterion:cuda()
		self.withCUDA = true
	else
		self.withCUDA = false
	end

	local x, dl_dx = self.mlp:getParameters()

	self.bestModel = nil
	self.bestError = 2^20
	print("# ".. self.name .. ": training")

	function doTrain(module, reg, criterion, config, trainDataset, currentEpoch)
		module:training()


		-- Create a batched iterator
		local trainIter = batchIterator:initialize({
			data = trainDataset,
			randomize = true,
			cuda = self.withCUDA,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs/2}
		})

		local nextBatch = trainIter:nextBatch()

		print("# Epoch " .. currentEpoch)
		local trainError = 0

		-- one epoch
		while (nextBatch ~= nil) do

			-- create eval clojure
			local feval = function(x_new)
				collectgarbage()

				-- get the new parameters
				if x ~= x_new then
					x:copy(x_new)
				end

				-- reset gradients
				dl_dx:zero()

				-- evaluate on complete mini-batch
				
				-- forward pass
				local outputs = module:forward(nextBatch.inputs)
				-- print(outputs:size())

				local pred, tar

				if (config.criterion == 'cosine') then
					pred = {outputs, nextBatch.targets}
					if (self.withCUDA == true) then
						tar = torch.Tensor(nextBatch.currentBatchSize):fill(1):cuda()
					else
						tar = torch.Tensor(nextBatch.currentBatchSize):fill(1)
					end
				elseif (config.criterion == 'mse') then
					pred = outputs
					tar= nextBatch.targets
				end

				local loss_x = criterion:forward(pred, tar)
			    local back = criterion:backward(pred, tar)

			    local backward
				if (config.criterion == 'cosine') then
					backward = back[1]
				elseif (config.criterion == 'mse') then
					backward = back
				end

				-- backward pass
				module:backward(nextBatch.inputs, backward)

				-- return loss(x) amd dloss/dx
				return loss_x, dl_dx
			end

			-- optimize the current mini-batch
			if config.optimizer == 'adagrad' then
				local _, fs = optim.adagrad(feval, x, config.adagrad_config)
				trainError = trainError + fs[1]
				collectgarbage()
			elseif config.optimizer == 'sgd' then
				local _, fs = optim.adagrad(feval, x, config.sgd_config)
				trainError = trainError + fs.data
			else
				error('unknown optimization method')
			end

			nextBatch = trainIter:nextBatch()
		end

		-- report average error per epoch
		trainError = trainError/trainDataset:size()

		if (self.printParameters == true) then
			print("Parameters")
			print(module:getParameters())
		end
		collectgarbage()

		return trainError
	end


	self.epoch = 1
	self.extraIndex = 1
	local logger = optim.Logger(self.config.saveName, false) -- no timestamp
	logger.showPlot = false; -- if not run on a remote server, set this to true to show the learning curves in real time
	logger.plotRawCmd = 'set xlabel "Epochs"\nset ylabel "' .. self.config.criterion ..'"'
	logger.name = self.name

	self:visualize(0)

	while true do
		local itrainErr = doTrain(self.mlp, self.reg, self.criterion, self.config, self.trainDataset, self.epoch)
		collectgarbage()
		local trainErr = self:doTest(self.mlp, self.criterion, self.config, self.trainDataset)
		local devErr = self:doTest(self.mlp, self.criterion, self.config, self.devDataset)

		self:visualize(self.epoch)

		self.epoch = self.epoch + 1
		print('Train error:\t', string.format("%.10f", trainErr))
		print('Dev error:\t', string.format("%.10f", devErr))
		print('Best error:\t', string.format("%.10f (%d)", self.bestError, self.extraIndex))

		-- log the errors for plotting
		logger:add{['training error'] = trainErr, ['dev error'] = devErr}
		logger:style{['training error'] = '-', ['dev error'] = '-'}
		logger:plot()

		-- early stopping when the error on the test set ceases to decrease
		if (self.config.earlyStopping == true) then
			if (self.bestError - devErr > 1e-4) then
				self.bestError = devErr
				torch.save(self.config.saveName .. ".bin", self.mlp);

				self.extraIndex = 1
			else
				if (self.extraIndex < self.config.extraEpochs) then
					self.extraIndex = self.extraIndex + 1
				else
					print("# Composer: stopping - you have reached the maximum number of epochs after the best model")
					print("# Composer: best error: ", string.format("%.4f", self.bestError))
					break					 
				end
			end
		end
	end
 end

 function CompositionModel:predict(onDev, onTest, onFull, cmhDictionary, devSet, testSet, fullSet)

	if (self.config.gpuid >= 0) then
		self.withCUDA = true
	else
		self.withCUDA = false
	end

	print("iterator type " .. self.iteratorType)
	if (self.iteratorType == 'table') then
		batchIterator = require 'iterators.TableBatchIterator'
	elseif (self.iteratorType == 'quad') then
		batchIterator = require 'iterators.QuadBatchIterator'
	end

 	function predict(module, config, dataset)
 		module:evaluate()
 		local predictions = torch.Tensor(dataset:size(), dataset[1][2]:size()[1])

 		-- one pass through the data
		local dataIter = batchIterator:initialize({
			data = dataset,
			randomize = false,
			cuda = self.withCUDA,
			batchSize = config.batchSize,
			inputShape = {2, self.inputs/2}
		})

		local t = 1
		local nextBatch = dataIter:nextBatch()

		while (nextBatch ~= nil) do
			local preds = module:forward(nextBatch.inputs):float()
			predictions[{{t, t + nextBatch.currentBatchSize - 1}, {}}] = preds
			t = t + nextBatch.currentBatchSize

			nextBatch = dataIter:nextBatch()
		end
		return predictions
 	end

	function savePredictions(cmhDictionary, predictions, saveName, indexSet, separator)
		local field_delim = separator or ' '
		local outputFileName = saveName .. '.pred'
		local f = io.open(outputFileName, "w")
		f:write(string.format("%d %d\n", predictions:size()[1], predictions:size()[2]))

		for i = 1, predictions:size()[1] do
			local cidx = indexSet[i][3]
			local word = cmhDictionary[cidx]

			f:write(word .. field_delim)
			for j = 1,predictions:size()[2] do
				f:write(string.format("%.12f", predictions[i][j]))
				if (j < predictions:size()[2]) then
				  f:write(field_delim)
				end
			end
			f:write("\n")
		end
		f:close()
	end

	print(" # Loading best model from " .. self.config.saveName)
	self.mlp = nil
	self.config.adagrad_config=nil
	print(self)
	collectgarbage()
	print("freeMemory ", freeMemory)
	print("totalMemory", totalMemory)
	local bestModel = torch.load(self.config.saveName .. ".bin")



 	if (onDev == true) then
		print(" # Creating dev set predictions... ")
		local devPredictions = predict(bestModel, self.config, self.devDataset)
		local devLoss = self:doTest(bestModel, self.criterion, self.config, self.devDataset)
		print(string.format("# dev loss: %.4f", devLoss))
		savePredictions(cmhDictionary, devPredictions, self.config.saveName .. '_dev', devSet, ' ')
	end

	if (onTest == true) then
		print(" # Creating test set predictions... ")
		local testPredictions = predict(bestModel, self.config, self.testDataset)
		local testLoss = self:doTest(bestModel, self.criterion, self.config, self.testDataset)
		print(string.format("# test loss: %.4f", testLoss))
		savePredictions(cmhDictionary, testPredictions, self.config.saveName .. '_test', testSet, ' ')
	end

	if (onFull == true) then
		print(" # Creating full predictions... ")
		local fullPredictions = predict(bestModel, self.config, self.fullDataset)
		savePredictions(cmhDictionary, fullPredictions, self.config.saveName .. '_full', fullSet, ' ')
	end
end

