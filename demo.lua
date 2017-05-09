require 'rnn'
require 'nn'
require 'optim'
require 'gnuplot'
require 'util.checkout_rnnKernel'
require 'util.encoder'
require 'io.csp_io'

na = 19
nm = 1
nr = 30
m = 1
model2d = torch.load('model2d.dat')

-- build autoencoder
print('build encoder...')
encoder = nn.Sequential() -- autoencoder net
for i = 1,11 do
        encoder:add(model2d:get(i))
end
encoder = nn.Sequencer(encoder)
print(encoder)

-- build decoder
print('build decoder...')
decoder = nn.Sequential()
for i = 12,22 do
        decoder:add(model2d:get(i))
end
decoder = nn.Sequencer(decoder)
print(decoder)

-- build pre-trainset and trainset
print('build pre-trainset and trainset')
csp_inputs = {}
csp_targets = {}
encoded_inputs = {}
encoded_targets = {}
assert(path.exists('rnn_trainset.dat'),'rnn_trainset.dat does not exist')
local tmp = torch.load('rnn_trainset.dat')
csp_inputs = tmp[1]
csp_targets = tmp[2]
encoded_inputs = tmp[3]
encoded_targets = tmp[4]

--build validation set
local tmp = torch.load('valset.dat')
val_csp_inputs = tmp[1]
val_csp_targets = tmp[2]

--build preRNN
rnnKernel = torch.load('rm.dat')
preRNN = nn.Sequencer(rnnKernel)
print(preRNN)
