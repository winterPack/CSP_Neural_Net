require 'torch';
require 'nn';
require 'gnuplot';
require 'csvigo';

function read_coarse_grained_csv(angle,runId,material)
    -- To-do list:
    --  make data_dir an optional variable
    --  codes for "material"
    local data_dir, material_labels, csv_file,csv,csv_tensor
    angle=(angle-1)*5
    data_dir= 'tilt_pts_csp_50x50x50'
    material_labels = {'Cu','Ag'}
    csv_file = string.format("%s/tilt%d_run%d.csv",data_dir,tostring(angle),tostring(runId))
    csv_file = paths.concat(csv_file)
    assert(csv_file,'invalid csv_file:' .. csv_file)
    csv = csvigo.load{path = csv_file, verbose = false, mode = "raw"}
    csv_tensor = torch.Tensor(csv)
    csv_tensor = csv_tensor:reshape(50,50,50)
    return csv_tensor
end

na = 19;
nm = 1;
nr = 30;

print('create training set')

csp_data = torch.Tensor(na,nm,nr,50,50,50)
label_data = torch.Tensor(na,nm,nr)

print('loading csv...')
material = 1
for angle = 1, na do
    print('processing angle ' .. angle)
    for runId = 1, nr do
        csp_data[angle][material][runId] = read_coarse_grained_csv(angle,runId,material)
        label_data [angle][material][runId] = angle 
    end
end
print('loading finished')
trainset = {}
trainset.input = csp_data:reshape(na*nm*nr*50,1,50,50)

-- The trainer is expecting trainset[123] to be {data[123], label[123]}
setmetatable(trainset, 
{__index = function(t, i) 
    return {t.input[i], t.input[i]}  
    end}
);

function trainset:size()
    return self.input:size(1)
end


print('create validation set')

na = 19
nm = 1
nr = 20
csp_data = torch.Tensor(na,nm,nr,50,50,50)
label_data = torch.Tensor(na,nm,nr)
print('loading csv...')
material = 1
for angle = 1, na do
    print('processing angle ' .. angle)
    for runId = 31, 40 do
        csp_data[angle][material][runId-30] = read_coarse_grained_csv(angle,runId,material)
        -- label_data [angle][material][runId-30] = angle 
    end
end
print('loading finished')
valset = {}
valset.input = csp_data:reshape(na*nm*nr*50,1,50,50)
setmetatable(valset,
    {__index = function(t,i)
        return {t.input[i],t.input[i]}
    end
    })

function valset:size()
    return self.input:size(1)
end

print('build model')
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


-- Cost function/ criterion
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1 


print('training')
-- training
for i = 1,200 do
    print('step ' .. i)
    trainer:train(trainset)
    local currentError = 0
    for t = 1,valset:size() do
         local input = valset[t][1]
         local target = valset[t][2]
         currentError = currentError + criterion:forward(model:forward(input), target)
     end
     currentError = currentError/valset:size()
     print('  valset MSE = '..currentError)
     if (i%10 == 1) then
         snap_file = 'model2d.' .. i
         print(snap_file .. ' saved')
         torch.save(snap_file,model)
     end
end

torch.save('model2d.dat',model)
