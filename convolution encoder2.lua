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

csp_data = torch.Tensor(na,nm,nr,50,50,50)
label_data = torch.Tensor(na,nm,nr)

if (path.exists("/Users/winter/Documents/th/CSP/trainset.dat")) then
    print('load "trainset.dat"') -- load previous dat file
    torch.load('trainset.dat')
else
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
    trainset.input = csp_data:reshape(na*nm*nr,1,50,50,50)
    trainset.target = trainset.input:clone()

    -- The trainer is expecting trainset[123] to be {data[123], label[123]}
    trainset.mt = {}
    trainset.mt.__index = function(t,key) 
	return {t.input[key],t.target[key]}
    end
    setmetatable(trainset,trainset.mt)

    function trainset:size()
        return self.input:size(1)
    end

end


print('build model')
model = nn.Sequential()

model:add(nn.VolumetricConvolution(1,5,6,6,6)) -- 5 x 45x45x45
model:add(nn.ReLU())
maxPool1 = nn.VolumetricMaxPooling(5,5,5) -- 5x 9x9x9
model:add(maxPool1)

model:add(nn.Reshape(5*9*9*9))

model:add(nn.Linear(5*9*9*9,200))
--above is encoder
--below is decoder
model:add(nn.Linear(200,5*9*9*9))

model:add(nn.Reshape(5,9,9,9))

model:add(nn.VolumetricMaxUnpooling(maxPool1))
model:add(nn.ReLU())
model:add(nn.VolumetricFullConvolution(5,1,6,6,6)) -- 5 x 45x45x45

-- Test model with a forawrd()
-- model:forward(torch.Tensor(1,50,50,50):zero()):size()


-- Cost function/ criterion
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 200 -- do 200 epochs of training

-- training
trainer:train(trainset)

torch.save('model.dat',model)
