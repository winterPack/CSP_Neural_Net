function circular_shift(data,x,y)
	local szX = data:size(2)
	local szY = data:size(3)
	local idxs = torch.LongTensor():range(1,szX):reshape(szX,1)
	idxs = (idxs-x-1)%szX + 1
	idxs = torch.expand(idxs,szX,szY)
	-- print(idxs)
	local out = data:view(szX,szY)
	out = out:gather(1,idxs)
	idxs = torch.LongTensor():range(1,szY):reshape(1,szY)
	idxs = (idxs-y-1)%szY + 1
	idxs = torch.expand(idxs,szX,szY)
	out = out:gather(2,idxs)
	return out:view(1,szX,szY)
end