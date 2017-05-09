require 'nn';
require 'rnn'; -- for sequencer
require 'optim';
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/csp_io.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_chunks.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_shuffle.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_flatten.lua')
dofile('../io/circular_shift.lua' )

local PRE_TRAIN = true

print ('load train set and validation set')
csp_inputs,csp_targets,val_csp_inputs,val_csp_targest = load_csp_data()

nBatch = 200

inputs = get_flatten(csp_inputs)
-- local tmp = {}
-- for i = 1,250 do
--     table.insert(tmp, inputs[i])
-- end
-- inputs = tmp

function feval(params)
    preTrainModel:zeroGradParameters()
    local outputs = preTrainModel:forward(batchInputs)
    local loss = criterion:forward(outputs,batchInputs)
    -- print({epoch,loss})
    local dloss_doutputs = criterion:backward(outputs,batchInputs)
    preTrainModel:backward(batchInputs,dloss_doutputs)
    return loss,gradParams
end

criterion = nn.SequencerCriterion(nn.MSECriterion())


if (PRE_TRAIN) then
	print('build model')
    model = nn.Sequential()

    model:add(nn.SpatialConvolution(1,16,5,5)) 
    model:add(nn.ReLU())
    maxPool1 = nn.SpatialMaxPooling(3,3)
    model:add(maxPool1)

    model:add(nn.SpatialConvolution(16,16,5,5))
    model:add(nn.ReLU())
    maxPool2 = nn.SpatialMaxPooling(3,3)
    model:add(maxPool2)

    --above is encoder
    -- model:add(nn.Reshape(8*5*5))
    -- model:add(nn.Reshape(8,5,5))
    --below is decoder

    model:add(nn.SpatialMaxUnpooling(maxPool2))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(16,16,5,5))

    model:add(nn.SpatialMaxUnpooling(maxPool1))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(16,1,5,5))
    model:add(nn.ReLU())

    -- model = nn.Sequencer(model)
    print(model)

    layers = {{1,2,3,10,11,12,13},{4,5,6,7,8,9}}
    encoderLayers = {{1,2,3},{4,5,6}}
    maxEpoch = 100;
    batchSize = 5;
    nBatch = math.ceil(#inputs/batchSize)
    idxs = {}
    for i = 1,#inputs do
    	table.insert(idxs,i)
    end
    print('pre-train model')
    for i = 1,#layers do
    	preTrainModel = nn.Sequential()
        encoderModel = nn.Sequential()
        for j = 1,#(encoderLayers[i]) do
            encoderModel:add(model:get(encoderLayers[i][j]))
        end
    	for j = 1,#(layers[i]) do
    		preTrainModel:add(model:get(layers[i][j]))
    	end
    	preTrainModel = nn.Sequencer(preTrainModel)
        encoderModel = nn.Sequencer(encoderModel)
        params, gradParams = preTrainModel:getParameters()
        next_inputs = {}

        optimState = {learningRate = 0.001}
    	for epoch = 1,maxEpoch do
    		local epoch_loss = 0
    		local count = 0
    		batch_Idxs = get_chunks(get_shuffle(idxs),nBatch)
    		for j = 1,nBatch do
    			batchInputs = {}
    			for k = 1,#batch_Idxs[j] do
    				local x = math.random(50)
    				local y = math.random(50)
    				local tmp = inputs[batch_Idxs[j][k]];
    				table.insert(batchInputs, tmp)
    			end
    			_,loss = optim.rmsprop(feval,params,optimState)
    			-- str = string.format('\r  batchId = %4d, loss = %8.4f',j,loss[1]/#batchInputs)
    			-- print(str)
    			epoch_loss = epoch_loss + loss[1]
    			count = count + #batchInputs
                if (epoch == maxEpoch) then
                    local tmp = encoderModel:forward(batchInputs)
                    for k = 1, #tmp do
                        table.insert(next_inputs, tmp[k]:clone())
                    end
                end
    		end
            if (epoch == maxEpoch) then
                inputs = next_inputs
            end
    		epoch_loss = epoch_loss/count
    		-- logger:add{epoch_loss}
    		print(string.format('epoch: %4d, epoch_loss = %8.2f',epoch,epoch_loss))
    	end
    end
end
torch.save('autoencoder2d_version_2.dat',model)

print('full training')
model = nn.Sequencer(model) -- make it a sequencer for mini-batch training
inputs = get_flatten(csp_inputs)
params, gradParams = model:getParameters()

function feval(params)
    model:zeroGradParameters()
    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs,batchInputs)
    -- print({epoch,loss})
    local dloss_doutputs = criterion:backward(outputs,batchInputs)
    model:backward(batchInputs,dloss_doutputs)
    return loss,gradParams
end

startEpoch = 1
maxEpoch = 10000
batchSize = 5;
local idxs = {}
for i = 1,#inputs do
	table.insert(idxs,i)
end
nBatch = math.ceil(#inputs/batchSize)
for epoch = startEpoch,maxEpoch do
	epoch_loss = 0
	count  = 0
    batch_Idxs = get_chunks(get_shuffle(idxs),nBatch)
    for i = 1,nBatch do
    	batchInputs = {}
    	for j = 1,#(batch_Idxs[i]) do
	  --   	local x = math.random(50)
			-- local y = math.random(50)
    		local tmp = inputs[batch_Idxs[i][j]]
    		table.insert(batchInputs,tmp)
    	end
    	_,loss = optim.rmsprop(feval,params)
    	epoch_loss = epoch_loss + loss[1]
		count = count + #batchInputs
    end
    epoch_loss = epoch_loss/count
    print(string.format('epoch: %4d, epoch_loss = %8.2f',epoch,epoch_loss))
    if (epoch%10 == 0) then
		bak_file = string.format('../bak_cache/autoencoder2d_version_2.%04d.dat',epoch)
		torch.save(bak_file,model)
    end
end
torch.save('autoencoder2d_version_2.dat',model)
