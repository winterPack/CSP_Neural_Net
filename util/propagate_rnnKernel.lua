require 'rnn'
require 'gnuplot'
require 'csvigo'

function propagate_rnnKernel(inputs,targets,rnnKernel,a,m,r,layerId)
	local model2d = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/model2d.dat')
	local model = nn.Sequential()
	for i = 1,11 do
		model:add(model2d:get(i))
	end
	model:add(rnnKernel)
	for i = 12,22 do
		model:add(model2d:get(i))
	end
	model:forget()
	model:evaluate()
	local na = 19
	local nm = 1
	local nr = #inputs/na/nm
	local idx = (a-1)*nr+r
	local inputLayer = inputs[idx][layerId]
	local outputLayer = inputLayer
	local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	gnuplot.raw('set multiplot layout 3,5 title "propagate_rnnKernel:'..title..'" font ",20"')
	gnuplot.raw('set bmargin 2')
	for i = 1,15 do
		outputLayer = model:forward(outputLayer):clone()
		if (i%1 == 0) then
			gnuplot.imagesc(outputLayer:reshape(50,50))
		end
	end

end

function sleep(s)
  local ntime = os.clock() + s
  repeat until os.clock() > ntime
end

function propagate_rnnKernel_reference(inputs,targets,rnnKernel,a,m,r,layerId)
	local model2d = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/model2d.dat')
	local model = nn.Sequential()
	for i = 1,11 do
		model:add(model2d:get(i))
	end
	model:add(rnnKernel)
	for i = 12,22 do
		model:add(model2d:get(i))
	end
	model:forget()
	model:evaluate()
	local na = 19
	local nm = 1
	local nr = #inputs/na/nm
	local idx = (a-1)*nr+r
	local inputLayer = inputs[idx][layerId]
	local outputLayer = inputLayer
	local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	gnuplot.raw('set multiplot layout 3,5 title "propagate_rnnKernel_reference:'..title..'" font ",20"')
	gnuplot.raw('set bmargin 2')
	for i = 1,15 do
		-- outputLayer = model:forward(outputLayer):clone()
		if (i%1 == 0) then
			local k = (layerId+i-1)%50+1
			-- gnuplot.imagesc(outputLayer:reshape(50,50))
			gnuplot.imagesc(targets[idx][k]:view(50,50))
		end
	end

end

function propagate_rnnKernel_tofile(inputs,targets,rnnKernel,a,m,r,layerId,Nlayers,filename)
	local Nlayers = Nlayers or 50
	local filename = filename or 'out.csv'
	local model2d = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/model2d.dat')
	local model = nn.Sequential()
	for i = 1,11 do
		model:add(model2d:get(i))
	end
	model:add(rnnKernel)
	for i = 12,22 do
		model:add(model2d:get(i))
	end
	model:forget()
	model:evaluate()
	local na = 19
	local nm = 1
	local nr = #inputs/na/nm
	local idx = (a-1)*nr+r
	local inputLayer = inputs[idx][layerId]
	local outputLayer = inputLayer
	-- local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	-- gnuplot.raw('set multiplot layout 3,5 title "propagate_rnnKernel:'..title..'" font ",20"')
	-- gnuplot.raw('set bmargin 2')
	local csv_data = {}
	print (Nlayers)
	for i = 1, Nlayers do
		outputLayer = model:forward(outputLayer):clone()
		local tmp = {}
		for j = 1,50 do
			for k = 1,50 do
				table.insert(tmp,outputLayer[1][j][k])
			end
		end
		table.insert(csv_data,tmp)
		-- if (i%1 == 0) then
		-- 	gnuplot.imagesc(outputLayer:reshape(50,50))
		-- end
	end
	csvigo.save{path = filename, data = csv_data}
end