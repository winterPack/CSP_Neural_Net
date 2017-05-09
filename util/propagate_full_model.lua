require 'nn'
require 'rnn'
require 'gnuplot'
require 'csvigo'

function propagate_full_model(csp_inputs,a,m,r,layerId,Nlayers,model_file)
	local Nlayers  = Nlayers or 15
	local model_file = 'full_model.dat'
	local model = torch.load(model_file)
	model:evaluate()
	model:forget()
	model = model:get(1)
	local nr = #csp_inputs/19
	local idx = (a-1)*nr+r
	local outputLayer = csp_inputs[idx][layerId]
	local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	gnuplot.raw('set multiplot layout 3,5 title "propagate_rnnKernel_reference:'..title..'" font ",20"')
	gnuplot.raw('set bmargin 2')
	for i = 1,15 do
		outputLayer = model:forward(outputLayer):clone()
		gnuplot.imagesc(outputLayer:reshape(50,50))
	end
end

function propagate_full_model_to_file(csp_inputs,a,m,r,layerId,Nlayers,model_file,output_csv_file)
	local Nlayers  = Nlayers or 15
	local model_file = 'full_model.dat'
	local model = torch.load(model_file)
	local output_csv_file = output_csv_file or 'out.csv'
	model:evaluate()
	model:forget()
	model = model:get(1)
	local nr = #csp_inputs/19
	local idx = (a-1)*nr+r
	local outputLayer = csp_inputs[idx][layerId]
	-- local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	-- gnuplot.raw('set multiplot layout 3,5 title "propagate_rnnKernel_reference:'..title..'" font ",20"')
	-- gnuplot.raw('set bmargin 2')
	local csv_data = {}
	for i = 1,100 do
		outputLayer = model:forward(outputLayer):clone()
		-- gnuplot.imagesc(outputLayer:reshape(50,50))
		local tmp = {}
		for j = 1,50 do
			for k = 1,50 do
				table.insert(tmp,outputLayer[1][j][k])
			end
		end
		table.insert(csv_data,tmp)
	end
	-- return csv_data
	csvigo.save{path = output_csv_file, data = csv_data}
end