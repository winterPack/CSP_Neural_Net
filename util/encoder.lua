require 'rnn'
require 'gnuplot'

function show_image(inputs,model,a,m,r,layerId)
	local na = 19
	local nm = 1
	local nr = #inputs/na/nm
	local idx = (a-1)*nr+r
	local output = model:forward(inputs[idx])
	local title = string.format('angle=%2d,Cu,runId=%2d,layerId=%2d',(a-1)*5,r,layerId)
	gnuplot.raw('set multiplot layout 1,2 title "2D convolutional encoder:'..title..'" font ",20"')
	gnuplot.raw('set bmargin 2')
	-- gnuplot.raw('set palette color')
	-- gnuplot.raw('pm3d map')
	gnuplot.raw('set title "input"')
	gnuplot.imagesc(inputs[idx][layerId]:reshape(50,50))	
	gnuplot.raw('set title "encoder NN"')
	gnuplot.imagesc(output[layerId]:reshape(50,50))
	gnuplot.plotflush()
end