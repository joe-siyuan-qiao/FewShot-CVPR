require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'image'

local DataSampler = torch.class('DataSampler')

--------------------------------------------------------------------------------
-- function: init
function DataSampler:__init(config, split, ftpool, mapping)
  assert(split == 'train' or split == 'val')
  self.split = split
  self.datadir = config.datadir
  self.mapping = ((mapping == nil) and torch.load(config.mpdir)) or mapping

  if ftpool then
    self.ftpool = ftpool
  else
    self.ftpool = torch.load(config.ftdir)
  end
  self.indvinp = self.ftpool.input.indv
  self.meaninp = self.ftpool.input.mean
  self.label = self.ftpool.label

  self.id2class = {}
  for class, _ in pairs(self.indvinp) do
    table.insert(self.id2class, class)
  end
  self.id2class = self.mapping.id2class

  -- misc
  if split == 'train' then
    self.__size = config.maxload
  elseif split == 'val' then
    self.__size = config.testmaxload
  end

  torch.manualSeed(config.seed)
  math.randomseed(config.seed)

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: get a sample
function DataSampler:get(head)
  local inputs, labels = self:sampling(head)
  return inputs, labels
end

--------------------------------------------------------------------------------
-- function: get size of epoch
function DataSampler:size()
  return self.__size
end

--------------------------------------------------------------------------------
-- function: sampling
local labels = torch.LongTensor(80, 1)
local inputs = torch.FloatTensor(160, 64)
function DataSampler:sampling(head)
  for i = 1, 80 do
    labels[i] = i
    local class = self.id2class[i]
    if math.random() > 0.3 then
      local inpid = torch.random(#self.indvinp[class])
      inputs:narrow(1,i,1):copy(self.indvinp[class][inpid][1])
    else
      inputs:narrow(1,i,1):copy(self.meaninp[class][1][1])
    end
    inpid = torch.random(#self.indvinp[class])
    inputs:narrow(1,80+i,1):copy(self.indvinp[class][inpid][1])
  end
  return inputs, labels
end
