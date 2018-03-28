require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'

local WeightNet, _ = torch.class('nn.WeightNet', 'nn.Container')

--------------------------------------------------------------------------------
-- function: constructor
function WeightNet:__init(config)
  self.batch = config.batch
  local net = nn.Sequential()

  local n = nn:Identity()

  net:add(n):add(nn.Normalize(2)):add(nn.Linear(64, 64))
     :add(nn.ReLU()):add(nn.Linear(64, 64)):add(nn.Normalize(2))
  net:get(3).bias:zero()
  net:get(3).weight:zero()
  net:get(5).bias:zero()
  net:get(5).weight:zero()
  for i= 1, 64 do
    net:get(3).weight[i][i] = 1.0
    net:get(5).weight[i][i] = 1.0
  end
  self.net = net:float()

  self.linear = nn.Sequential():add(nn.Linear(64, 80, false))
end

--------------------------------------------------------------------------------
-- function: training
function WeightNet:training()
  self.net:training()
  self.linear:training()
end

--------------------------------------------------------------------------------
-- function: evaluate
function WeightNet:evaluate()
  self.net:evaluate()
  self.net:evaluate()
end

--------------------------------------------------------------------------------
-- function: to cuda
function WeightNet:cuda()
  self.net:cuda()
  self.linear:cuda()
end

--------------------------------------------------------------------------------
-- function: to float
function WeightNet:float()
  self.net:float()
  self.linear:float()
end

--------------------------------------------------------------------------------
-- function: clone
function WeightNet:clone(...)
  local f = torch.MemoryFile("rw"):binary()
  f:writeObject(self); f:seek(1)
  local clone = f:readObject(); f:close()

  if select('#', ...) > 0 then
    clone.net:share(self.net, ...)
    clone.linear:share(self.linear, ...)
  end

  return clone
end

return nn.WeightNet
