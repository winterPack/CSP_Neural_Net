require 'rnn';
require 'nn';
require 'optim';
require 'io.csp_io';
require 'io.get_shuffle';

csp_inputs,csp_targets,val_csp_inputs,val_csp_targets = load_csp_data()

print('generate encoded data')
encoded_inputs = {}
encoded_targets = {}
model2d = torch.load('model2d.dat')
encoder = nn.Sequential()
for i = 1,11 do
	encoder:add(model2d:get(i))
end
for i = 1,#csp_inputs do
	local ins = {}
	local tars = {}
	for j = 1,50 do
		table.insert(ins,encoder:forward(csp_inputs[i][j]):clone())
		table.insert(tars,encoder:forward(csp_targets[i][j]):clone())
	end
	table.insert(encoded_inputs,ins)
	table.insert(encoded_targets,tars)
end

local tmpConfig = {}
encoded_inputs = get_shuffle(encoded_inputs,tmpConfig)
encoded_targets = get_shuffle(encoded_targets,tmpConfig)

print('build rnn kernel')
rm1 = nn.Recurrent(200,nn.Identity(),nn.Linear(200,200),nn.Sigmoid(),5)
rm2 = nn.Recurrent(200,nn.Sigmoid(),nn.Linear(200,200),nn.Sigmoid(),5)
modelRNN = nn.Sequential()
	:add(rm1)
	:add(nn.Linear(200,200))
	-- :add(rm2)
	-- :add(nn.Linear(200,200))
modelRNN = nn.Sequencer(modelRNN)

params, gradParams = modelRNN:getParameters()
criterion = nn.SequencerCriterion(nn.MSECriterion())

function feval(params)
	gradParams:zero()
	local outputs = modelRNN:forward(batchInputs)
	local loss = criterion:forward(outputs,batchTargets)
	local dloss_doutputs = criterion:backward(outputs, batchTargets)
      	modelRNN:backward(batchInputs, dloss_doutputs)
	return loss,gradParams
end

print('pre-train model')
maxEpoch = 1000
optimState = {learningRate = 0.0001, momentum = 0.8}
modelRNN:training()
for epoch = 1,maxEpoch do 
    local epoch_loss = 0
    local count = 0
    for i = 1,#encoded_inputs do
	local tmpConfig = {}
	batchInputs = encoded_inputs[i]
	batchTargets = encoded_targets[i]
	-- batchInputs = get_shuffle(batchInputs,tmpConfig)
	-- batchTargets = get_shuffle(batchTargets,tmpConfig)
	modelRNN:forget()
	_,loss = optim.rmsprop(feval,params,optimState)
--	_,loss = optim.sgd(feval,params,optimState)
	epoch_loss = epoch_loss +loss[1]
	count = count+#batchInputs
--	print(string.format('  batchID: %4d,  loss = %8.2f',i,loss[1]/#batchInputs))
    end
    epoch_loss = epoch_loss/count
    print(string.format('epoch: %4d, epoch_loss = %8.2f',epoch,epoch_loss))
    if (epoch%10 == 0) then
	bak_file = string.format('bak_cache/modelRNN.tight.%04d.dat',epoch)
	torch.save(bak_file,modelRNN)
    end
end
torch.save('modelRNN.tight.dat',modelRNN)
