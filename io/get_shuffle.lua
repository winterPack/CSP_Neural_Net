require 'torch';
function get_shuffle(data,config)
	local config = config or {}
	config.perm_list = config.perm_list or torch.randperm(#data)
	local perm_list = config.perm_list
	local out = {}
	for i = 1,#data do
		table.insert(out,data[perm_list[i]])
	end
	return out
end
