require 'torch'
require 'csvigo'
require 'nn'

function read_coarse_grained_csv(angle,runId,material)
    -- To-do list:
    -- make data_dir an optional variable
    -- codes for "material"
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

-- function load_trainset(nm,nr)
--     nm = (nm==nil) and 1 or nm
--     nr = (nr==nil) and 30 or nr
--     assert(nm == 1, 'Doh! load_trainset is not implemented for 2 materials yet')
--     print(string.format('load_trainset(%2d,%2d)...',nm,nr))
--     local data = {}
--     local labels = {}
--     for a = 1,19 do
--         for r = 1,nr do
--             for m = 1,nm do
--                 table.insert(data,read_coarse_grained_csv(a,r,m)):split(1,1)
--                 table.insert(labels,{a,r,m})
--             end
--         end
--     end
--     return data,labels
-- end

function load_csp_data()
    local tmp = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/rnn_trainset.dat')
    csp_inputs = tmp[1]
    csp_targets = tmp[2]
    tmp = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/valset.dat')
    val_csp_inputs = tmp[1]
    val_csp_targets = tmp[2]
    return csp_inputs,csp_targets,val_csp_inputs,val_csp_targets
end


function load_encoded_data(data)
    local data = data or '/share3/hydra_export/winter/CSP_Neural_Net/model2d.dat'
    local model2d = (torch.type(data) == 'string') and torch.load(data) or data
    assert(torch.type(model2d) == 'nn.Sequential', 'cannot load model')
    encoder = nn.Sequential()
    for i = 1,11 do
        encoder:add(model2d:get(i))
    end
    local csp_inputs,csp_targets,val_csp_inputs,val_csp_targets = load_csp_data()
    local encoded_inputs = {}
    local encoded_targets = {}
    for i = 1, #csp_inputs do
        local tmp_ins = {}
        local tmp_tars = {}
        for j = 1,#csp_inputs[i] do
            table.insert(tmp_ins, encoder:forward(csp_inputs[i][j]):clone())
            table.insert(tmp_tars, encoder:forward(csp_targets[i][j]):clone())
        end
        table.insert(encoded_inputs, tmp_ins)
        table.insert(encoded_targets, tmp_tars)
    end

    local val_encoded_inputs = {}
    local val_encoded_targets = {}
    for i = 1, #val_csp_inputs do
        local tmp_ins = {}
        local tmp_tars = {}
        for j = 1,#val_csp_inputs[i] do
            table.insert(tmp_ins, encoder:forward(val_csp_inputs[i][j]):clone())
            table.insert(tmp_tars, encoder:forward(val_csp_targets[i][j]):clone())
        end
        table.insert(val_encoded_inputs, tmp_ins)
        table.insert(val_encoded_targets, tmp_tars)
    end
    return encoded_inputs,encoded_targets,val_encoded_inputs,val_encoded_targets
end
