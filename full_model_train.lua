require 'nn'
require 'rnn'
require 'optim'
require 'io.csp_io'
require 'io.get_shuffle'

START_FROM_SCRATCH = nil

if (START_FROM_SCRATCH) then
	rnnKernel_file = 'bak_cache/modelRNN.LSTM.10000.dat';
	print('load pretrained rnn kernel')
	rnnKernel = torch.load(rnnKernel_file)
	print(rnnKernel)

	encoder2d_file =  'model2d.dat'
	print('load autoencoder')
	model2d = torch.load(encoder2d_file)
	print(rnnKernel)

	print('build full model')
	model = nn.Sequential()
	for i = 1,11 do
		model:add(model2d:get(i))
	end
	model:add(rnnKernel:get(1):get(1):get(1))
	model:add(rnnKernel:get(1):get(1):get(2))
	for i = 12,22 do
		model:add(model2d:get(i))
	end
	model = nn.Sequencer(model)
else
	model = torch.load('full_model.dat')
end
print(model)
print('load csp data')
csp_inputs,csp_targets,val_csp_inputs,val_csp_targest = load_csp_data()
-- extract a = 10 groups
csp_inputs10 = {}
csp_targets10 = {}
local a = 10
for r = 1,30 do
	local idx = (a-1)*30+r
	table.insert(csp_inputs10,csp_inputs[idx])
	table.insert(csp_targets10,csp_targets[idx])
end



params, gradParams = model:getParameters()
criterion = nn.SequencerCriterion(nn.MSECriterion())

function feval(params)
	gradParams:zero()
	local outputs = model:forward(batchInputs)
	local loss = criterion:forward(outputs,batchInputs)
	local dloss_doutputs = criterion:backward(outputs,batchTargets)
	model:backward(batchInputs,dloss_doutputs)
	return loss,gradParams
end

print('full train model')
logger = optim.Logger('full_model_train.log')
logger:setNames{'training_MSE'}
optimState = {learningRate = 0.0001, alpha = 0.97}
model:training()
batchSize = 40
startEpoch = 10001
maxEpoch = 15000

for epoch = startEpoch, maxEpoch do 
	local shuffle_config = {}
	csp_inputs10 = get_shuffle(csp_inputs10,shuffle_config)
	csp_targets10 = get_shuffle(csp_targets10,shuffle_config)
	local epoch_loss = 0
	local count = 0
	-- generate batchInputs
	model:forget()
	for i = 1,#csp_inputs10 do
		batchInputs = {}
		batchTargets = {}
		local batchStart = math.random(50)
		for j = 1,batchSize do
			local k = (batchStart+j-1)%50+1
			table.insert(batchInputs,csp_inputs10[i][k])
			table.insert(batchTargets,csp_targets10[i][k])
		end
		_,loss = optim.rmsprop(feval,params,optimState)
		epoch_loss = epoch_loss +loss[1]
		count = count+#batchInputs
	end
	epoch_loss = epoch_loss/count
	logger:add{epoch_loss}
	print(string.format('epoch: %4d, epoch_loss = %8.2f',epoch,epoch_loss))
	if (epoch%10 == 0) then
		bak_file = string.format('bak_cache/full_model_train.%04d.dat',epoch)
		torch.save(bak_file,model)
    end
end
torch.save('full_model.dat',model)
