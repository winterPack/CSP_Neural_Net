function get_flatten(data)
	local out = {}
	for i = 1, #data do
		for j = 1,#(data[i]) do
			table.insert(out,data[i][j])
		end
	end
	return out
end