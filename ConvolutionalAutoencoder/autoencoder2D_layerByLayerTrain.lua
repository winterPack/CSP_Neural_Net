require 'nn';
require 'rnn'; -- for sequencer
require 'optim';
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/csp_io.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_chunks.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_shuffle.lua')
dofile('/share3/hydra_export/winter/CSP_Neural_Net/io/get_flatten.lua')

local USE_A_SMALL_SET = nil
local PRE_TRAIN = nil

na = 19;
nm = 1;
nr = 30;

local tmp = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/rnn_trainset.dat')
inputs = tmp[1]

-- shuffle inputs and groups into batches
nBatch = 200
local tmp = {}
for i = 1,#inputs do -- flatten data
    for j = 1,#(inputs[1]) do
        table.insert(tmp,inputs[i][j])
    end
end
tmp = get_shuffle(tmp)
if (USE_A_SMALL_SET) then
    inputs = tmp  -- use a smaller trainset for testing
    tmp = {}
    for i = 1,400 do
        table.insert(tmp,inputs[i])
    end
end

inputs = get_chunks(tmp,nBatch)

print('build valset')
tmp = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/valset.dat')
valset = get_flatten(tmp[1])
valset = get_chunks(valset,100)

print('build model')
if (PRE_TRAIN) then
    model = nn.Sequential()

    model:add(nn.SpatialConvolution(1,10,3,3))
    model:add(nn.ReLU())

    model:add(nn.SpatialConvolution(10,10,3,3)) 
    model:add(nn.ReLU())
    maxPool1 = nn.SpatialMaxPooling(2,2)
    model:add(maxPool1) -- 10x 23x23

    model:add(nn.SpatialConvolution(10,5,3,3))
    model:add(nn.ReLU())
    maxPool2 = nn.SpatialMaxPooling(2,2)
    model:add(maxPool2) -- 5x 10x10

    model:add(nn.Reshape(5*10*10))
    model:add(nn.ReLU())
    model:add(nn.Linear(5*10*10,200))
    --above is encoder
    --below is decoder
    model:add(nn.Linear(200,5*10*10))
    model:add(nn.ReLU())
    model:add(nn.Reshape(5,10,10))

    model:add(nn.SpatialMaxUnpooling(maxPool2))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(5,10,3,3))

    model:add(nn.SpatialMaxUnpooling(maxPool1))
    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(10,10,3,3))

    model:add(nn.ReLU())
    model:add(nn.SpatialFullConvolution(10,1,3,3))

    print(model)

    --pre-train
    criterion = nn.SequencerCriterion(nn.MSECriterion())
    -- logger=optim.Logger()
-- logger:setNames{'training_set_error'}
-- logger:style{'+-'}

    local maxIter = 5
    local preInputs = inputs
    local layers = {{1,2,3,4,5},{6,7,8},{9,10,11}}
    for i = 1,#layers do
        print(string.format('pretrain layers%d',i))
        preTrainModel = nn.Sequential()
        local encoder = nn.Sequential()
        for j = 1,#(layers[i]) do
            local k = layers[i][j]
            preTrainModel:add(model:get(k))
            encoder:add(model:get(k))
        end
        for j = #(layers[i]),1,-1 do
            local k = layers[i][j]
            preTrainModel:add(model:get(23-k))
        end
        preTrainModel = nn.Sequencer(preTrainModel)
        encoder = nn.Sequencer(encoder)
        print(preTrainModel)
        params, gradParams = preTrainModel:getParameters()
        optimState = {learningRate = 0.001,momentum = 0.8}
        local nextInputs = {}
        print ('pre-Training start')
        for epoch = 1,maxIter do
             epoch_error = 0 
             -- valset_loss =0
             count = 0
            for batchID = 1,#preInputs do

                local batchInputs = preInputs[batchID]
                function feval(params)
                    preTrainModel:zeroGradParameters()
                    local outputs = preTrainModel:forward(batchInputs)
                    local loss = criterion:forward(outputs,batchInputs)
                    -- print({epoch,loss})
                    local dloss_doutputs = criterion:backward(outputs,batchInputs)
                    preTrainModel:backward(batchInputs,dloss_doutputs)
                    return loss,gradParams
                end
                optim.sgd(feval,params,optimState)
                local outputs = preTrainModel:forward(batchInputs)
                local loss = criterion:forward(outputs,batchInputs)
                epoch_error = epoch_error+loss
                if (epoch == maxIter) then
                    table.insert(nextInputs,encoder:forward(batchInputs))
                end
                count = count + #batchInputs
                io.write(string.format('  batchId %6d; progress = %6.2f %%, error = %6.2f\n',batchID,100.0*batchID/#preInputs,loss/#batchInputs))
                io.flush()
            end
            -- logger:add{epoch_error}
            -- logger:plot()
            if (epoch == maxIter) then
                preInputs= nextInputs
            end
            -- for k = 1,100 do
            --     local valset_outputs = preTrainModel:forward(valset[k])
            --     valset_loss = criterion:forward(valset_outputs,valset[k]) + valset_loss
            -- end
            print(string.format('\nepoch = %4d, trainset_error = %8.4f',epoch,epoch_error/count))
        end
    end
    torch.save('model.preTrained.dat',model)
else
    model = torch.load('model.preTrained.dat')
    model = nn.Sequencer(model)
end

-- full parameter train
print('full trainging start')
params, gradParams = model:getParameters()
criterion = nn.SequencerCriterion(nn.MSECriterion())

function feval(params)
    model:zeroGradParameters()
    local outputs = model:forward(batchInputs)
    local loss = criterion:forward(outputs,batchInputs)
    -- print({epoch,loss})
    local dloss_doutputs = criterion:backward(outputs,batchInputs)
    model:backward(batchInputs,dloss_doutputs)
    return loss,gradParams
end

local preInputs = inputs

optimState = {learningRate = 0.001}
for epoch = 1, 100 do
    local total_loss = 0
    local count = 0
    for batchID = 1,#preInputs do
        batchInputs = preInputs[batchID]
        local _,loss = optim.rmsprop(feval,params,optimState)
        print(string.format('batchID = %4d, loss = %f',batchID,loss[1]/#batchInputs))
        total_loss = total_loss + loss[1]
        count = count + #batchInputs
    end
    print(string.format('epoch = %4d, loss = %6.2f',epoch,total_loss/count))
    if (epoch%10 == 0) then
        bak_file = string.format('model.2dencoder-%4d.dat',epoch)
        torch.save(bak_file,model)
    end
end

torch.save('model.2dencoder.dat',model)
