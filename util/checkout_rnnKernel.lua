require 'rnn'
require 'gnuplot'

function checkout_rnnKernel(inputs,targets,rnnKernel,a,m,r,layerId)
	local model2d = torch.load('/share3/hydra_export/winter/CSP_Neural_Net/model2d.dat')
	local model = nn.Sequential()
	for i = 1,11 do
		model:add(model2d:get(i))
	end
	model:add(rnnKernel)
	for i = 12,22 do
		model:add(model2d:get(i))
	end
	model = nn.Sequencer(model)
	local na = 19
	local nm = 1
	local nr = #inputs/na/nm
	local idx = (a-1)*nr+r
	local output = model:forward(inputs[idx])
	local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	gnuplot.raw('set multiplot layout 1,3 title "checkout_rnnKernel:'..title..'" font ",20"')
	gnuplot.raw('set bmargin 2')
	-- gnuplot.raw('set palette color')
	-- gnuplot.raw('pm3d map')
	gnuplot.raw('set title "input"')
	gnuplot.imagesc(inputs[idx][layerId]:reshape(50,50))	
	gnuplot.raw('set title "target"')
	gnuplot.imagesc(targets[idx][layerId]:reshape(50,50))
	gnuplot.raw('set title "NN generation"')
	gnuplot.imagesc(output[layerId]:reshape(50,50))
	gnuplot.plotflush()
end