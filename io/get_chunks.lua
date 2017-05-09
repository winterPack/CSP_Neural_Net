function get_chunks(data, nBatch)
	local nData = #data
	local out = {}
	assert(nData >= nBatch,'data size smaller than number of batches')
	local i = 1
	while (nBatch>0) do
		local chunk_size = math.floor(nData/nBatch)
		local tmp = {}
		for j = i,(i+chunk_size-1) do
			table.insert(tmp,data[j])
		end
		i = i+chunk_size
		table.insert(out,tmp)
		nBatch = nBatch-1
		nData = nData-chunk_size
	end
	return out
end