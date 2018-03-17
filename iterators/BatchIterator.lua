require 'torch'

local tablex = require('pl.tablex')
local BatchIterator = torch.class('torch.BatchIterator')

------------------------------------------------------
------------------------------------------------------
local BatchIterator = {}
BatchIterator.__index = BatchIterator

function BatchIterator:initialize(args)
end

function BatchIterator:init(args)
	local self = {}
	setmetatable(self, BatchIterator)

	self.data = {}
	for i = 1, tablex.size(args.data) do
		if (type(args.data[i]) ~= 'function') and (args.data[i] ~= nil) then
			table.insert(self.data, args.data[i])
		end
	end

	self.dataSize = tablex.size(self.data)

	self.batchSize = args.batchSize
	self.withCUDA = args.cuda

	self.indices = torch.range(1, self.dataSize, 1) -- go through indices linearly
	self.index = 0

	self.noInputs = args.noInputs
	self.inputSize = args.inputSize
	self.inputShape = {args.noInputs, args.inputSize}
	self.noOutputs = args.noOutputs

	self.pretrainedModel = args.pretrainedModel
	self.pretrainedModel2 = args.pretrainedModel2

	if (args.randomize == true) then
		-- shuffe indices; this means that each pass sees the training examples in a different order 
		self.indices = torch.randperm(self.dataSize, 'torch.LongTensor')
	end

	return self	
end

function BatchIterator:processExample(example)
	return {example[1]:clone(), example[2]}
end

function BatchIterator:nextBatch()
	if self.index >= self.dataSize then
		return nil
	end

	self.currentBatchSize = self.batchSize

	if (self.batchSize > self.dataSize - self.index) then
		self.currentBatchSize = self.dataSize - self.index
	end	

	local batch = {}

	local inputs = torch.Tensor(self.currentBatchSize, self.inputShape[1], self.inputShape[2])
	local targets = torch.Tensor(self.currentBatchSize)

	local k = 1
	for i = self.index + 1, self.index + self.currentBatchSize do
		-- load new sample
		local example = self.data[self.indices[i]]
		processed = self:processExample(example)

		inputs[k] = processed[1]
		targets[k] = processed[2]

		k = k + 1
	end

	self.index = self.index + self.currentBatchSize


	if (self.withCUDA) then
		batch = {inputs=inputs:cuda(), targets=targets:cuda(), currentBatchSize = self.currentBatchSize}
	else
		batch = {inputs=inputs, targets=targets, currentBatchSize = self.currentBatchSize}
	end

	return batch
end


function BatchIterator:nextBatchTable()
	if self.index >= self.dataSize then
		return nil
	end

	self.currentBatchSize = self.batchSize

	if (self.batchSize > self.dataSize - self.index) then
		self.currentBatchSize = self.dataSize - self.index
	end	

	local batch = {}
	local f_inputs = nil
	local f_targets = nil

	-- local inputs = torch.Tensor(self.noInputs, self.currentBatchSize, self.inputSize)
	local inputs = {}
	local targets = {}
	local words = {}

	for p = 1, self.noInputs do
		local input = nil
		if (self.data[1][1][p]:size()[1] == 1) then
			input = torch.Tensor(self.currentBatchSize)
		else
			input = torch.Tensor(self.currentBatchSize, self.data[1][1][p]:size()[1])
		end
		local k = 1
		for i = self.index + 1, self.index + self.currentBatchSize do
			input[k] = self.data[self.indices[i]][1][p]
			k = k + 1
		end
		table.insert(inputs, input)
	end

	for p = 1, self.noOutputs do
		local target = nil
		if (self.data[1][2][p]:size()[1] == 1) then
			target = torch.Tensor(self.currentBatchSize)
		else
			target = torch.Tensor(self.currentBatchSize, self.data[1][2][p]:size()[1])
		end
		local k = 1
		for i = self.index + 1, self.index + self.currentBatchSize do
			target[k] = self.data[self.indices[i]][2][p]
			k = k + 1
		end
		table.insert(targets, target)
	end

	if (self.withCUDA) then
		f_inputs = {}
		for key, tns in pairs(inputs) do
			table.insert(f_inputs, tns:cuda())
		end
		f_targets = {}
		for key, tns in pairs(targets) do
			table.insert(f_targets, tns:cuda())
		end
	else 
		f_inputs = inputs
		f_targets = targets
	end

	self.index = self.index + self.currentBatchSize

	batch = {inputs=f_inputs, targets=f_targets, currentBatchSize = self.currentBatchSize, words=words}

	return batch
end


------------------------------------------------------
------------------------------------------------------

return BatchIterator
