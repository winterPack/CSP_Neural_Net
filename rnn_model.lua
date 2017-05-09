require 'nn'
require 'rnn'
require 'io.csp_io'
require 'optim'
require 'util.encoder'
-- lr_pre = 0.01
-- lr_full = 0.0001
na = 19
nm = 1
nr = 30
m = 1
model2d = torch.load('model2d.dat')



-- build autoencoder
print('build encoder...')
encoder = nn.Sequential() -- autoencoder net
for i = 1,11 do
	encoder:add(model2d:get(i))
end
encoder = nn.Sequencer(encoder)
print(encoder)

-- build decoder
print('build decoder...')
decoder = nn.Sequential()
for i = 12,22 do
	decoder:add(model2d:get(i))
end
decoder = nn.Sequencer(decoder)
print(decoder)
-- build pre-trainset and trainset
print('build pre-trainset and trainset')
csp_inputs = {}
csp_targets = {}
encoded_inputs = {}
encoded_targets = {}
if (path.exists('rnn_trainset.dat')) then
	tmp = torch.load('rnn_trainset.dat')
	csp_inputs = tmp[1]
	csp_targets = tmp[2]
	encoded_inputs = tmp[3]
	encoded_targets = tmp[4]
else
	for a = 1,na do
		print('\tprocessing angle '..a..' ...')
		for r = 1,nr do
			local csv_data = read_coarse_grained_csv(a,r,m):split(1,1)
			local tmp = encoder:forward(csv_data) -- forward return a view
			encoded_data = {}
			for i = 1,#tmp do
				table.insert(encoded_data,tmp[i]:clone())
			end
			table.insert(csp_inputs,csv_data)
			table.insert(encoded_inputs,encoded_data)
		end
	end
	-- build targets
	for i = 1,na*nm*nr do
		local tmp1 = {}
		local tmp2 = {}
		for j = 1,50 do 
			local k = (j-1+1)%50+1
			table.insert(tmp1,csp_inputs[i][k])
			table.insert(tmp2,encoded_inputs[i][k])
		end
		table.insert(csp_targets,tmp1)
		table.insert(encoded_targets,tmp2)
	end

	torch.save('rnn_trainset.dat',{csp_inputs,csp_targets,encoded_inputs,encoded_targets})
end
-- build recurrent kernel
print ('build recurrent kernel')
rm = nn.Recurrent(500,
	nn.Linear(200,500),
	nn.Linear(500,500),
	nn.Sigmoid(),
	3)
rm = nn.Sequential()
	:add(rm)
	:add(nn.Linear(500,200))
preRNN = nn.Sequencer(rm)
print(preRNN)

preRNN:forget()
print('pretrain recurrent kernel')
pretrain_criterion = nn.SequencerCriterion(nn.MSECriterion())
params, gradParams = preRNN:getParameters()

optimState = {learningRate = 0.01, momentum = 0.9, maxIter = 200}
logger = optim.Logger('pretrainRNN.log')
logger:setNames{'training set error'}
logger:style{'+-'}

for epoch = 1,1000 do
	local epoch_error = 0
	for batchId = 1,na*nm*nr do
		local batchInputs = encoded_inputs[batchId]
		local batchTargets = encoded_targets[batchId]
		function feval(params)
			preRNN:zeroGradParameters()
			local outputs = preRNN:forward(batchInputs)
			local loss = pretrain_criterion:forward(outputs,batchTargets)
			local dloss_doutputs = pretrain_criterion:backward(outputs,batchTargets)
			preRNN:backward(batchInputs,dloss_doutputs)
			epoch_error = epoch_error+loss
			return loss,gradParams
		end
		optim.sgd(feval,params,optimState)
		local outputs = preRNN:forward(batchInputs)
		local loss = pretrain_criterion:forward(outputs,batchTargets)
		epoch_error = epoch_error+loss
	end
	if (epoch%10 == 0) then
		torch.save(string.format('preRNN-%04d.dat',epoch),preRNN)
	end
	epoch_error = epoch_error/na/nm/nr
	logger:add{epoch_error}
	print('epoch = '..epoch..', epoch_error = '..epoch_error)
	-- logger:plot()
end
torch.save('rm.dat',rm)
-- -- pretrain recurrent kernel
-- print('pretrain recurrent kernel')
-- pretrain_criterion = nn.SequencerCriterion(nn.MSECriterion())
-- local iteration = 1
-- while iteration < 100 do
-- 	local total_error = 0
-- 	for i = 1,na*nm*nr do
-- 	    preRNN:zeroGradParameters()
-- 	    local outputs = preRNN:forward(inputs[i])
-- 	    local err = pretrain_criterion:forward(outputs,targets[i])
-- 	    local gradOutputs = pretrain_criterion:backward(outputs,targets[i])
-- 	    local gradInputs = preRNN:backward(inputs[i],gradOutputs)
-- 	    preRNN:updateParameters(lr_pre)
-- 	    total_error = total_error+err
-- 	end
-- 	total_error = total_error/na/nm/nr
-- 	print(string.format('\titeration:%3d, MSE: %f',iteration,total_error))
-- 	iteration = iteration+1
-- end
-- torch.save('rm.dat',rm)

-- -- build trainset
-- print('build trainset')
-- data = {}
-- for a = 1,na do
-- 	print('\tprocessing angle '..a..' ...')
-- 	for r = 1,nr do
-- 		csv_data = read_coarse_grained_csv(a,r,m):split(1,1)
-- 		table.insert(data,csv_data)
-- 	end
-- end

-- inputs = data
-- targets = {}

-- for i = 1,na*nm*nr do
-- 	local tmp = {}
-- 	for j = 1,50 do 
-- 		local k = (j-1+1)%50+1
-- 		table.insert(tmp,inputs[i][k])
-- 	end
-- 	table.insert(targets,tmp)
-- end

-- -- build full RNN model
-- print('build full RNN model')
-- modelRNN = nn.Sequential()
-- for i = 1,11 do
-- 	modelRNN:add(model2d:get(i))
-- end
-- modelRNN:add(rm:get(1))
-- modelRNN:add(rm:get(2))
-- for i = 12,22 do
-- 	modelRNN:add(model2d:get(i))
-- end
-- modelRNN = nn.Sequencer(modelRNN)

-- -- train whole net
-- print('train whole net')
-- criterion = nn.SequencerCriterion(nn.MSECriterion())
-- modelRNN:forget()
-- local iteration = 1
-- while iteration < 50 do
-- 	local total_error = 0
-- 	for i = 1,na*nm*nr do
-- 	    modelRNN:zeroGradParameters()
-- 	    local outputs = modelRNN:forward(inputs[i])
-- 	    local err = criterion:forward(outputs,targets[i])
-- 	    local gradOutputs = criterion:backward(outputs,targets[i])
-- 	    local gradInputs = modelRNN:backward(inputs[i],gradOutputs)
-- 	    modelRNN:updateParameters(lr_full)
-- 	    total_error = total_error+err
-- 	end
-- 	total_error = total_error/na/nm/nr
-- 	print(string.format('\titeration:%3d, MSE: %8.4f',iteration,total_error))
-- 	iteration = iteration+1
-- end

-- torch.save('modelRNN.dat',modelRNN)