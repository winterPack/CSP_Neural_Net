require 'nn'
require 'rnn'
require 'io.csp_io'
require 'optim'
require 'util.encoder'

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
assert(path.exists('rnn_trainset.dat'),'rnn_trainset.dat does not exist')
local tmp = torch.load('rnn_trainset.dat')
csp_inputs = tmp[1]
csp_targets = tmp[2]
encoded_inputs = tmp[3]
encoded_targets = tmp[4]

--build preRNN
rm = torch.load('rm.dat')
preRNN = nn.Sequencer(rm)
print(preRNN)
pretrain_criterion = nn.SequencerCriterion(nn.MSECriterion())
params, gradParams = preRNN:getParameters()

optimState = {learningRate = 0.001, momentum = 0.8, maxIter = 20}
logger = optim.Logger('pretrainRNN.log')
logger:setNames{'training set error'}
logger:style{'+-'}

for epoch = 2001,3000 do
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
