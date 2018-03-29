local M = {}
local DataLoader = torch.class('DataLoader', M)

--------------------------------------------------------------------------------
-- function: create train/val data loaders
function DataLoader.create(config)
  local loaders = {}
  local ftpool = torch.load(config.ftdir)
  local mapping = torch.load(config.mpdir)
  for i, split in ipairs{'train', 'val'} do
    loaders[i] = M.DataLoader(config, split, ftpool, mapping)
  end

  return table.unpack(loaders)
end

--------------------------------------------------------------------------------
-- function: init
function DataLoader:__init(config, split, ftpool)
  torch.setdefaulttensortype('torch.FloatTensor')
  local seed = config.seed
  torch.manualSeed(seed)
  if DataSampler then local donothing=true else
  if config.fft then paths.dofile('DataSamplerFFt.lua')
  else paths.dofile('DataSampler.lua') end end
  self.ds = DataSampler(config, split, ftpool)
  local sizes = self.ds:size()
  self.__size = sizes
  self.batch = config.batch
  self.hfreq = config.hfreq
end

--------------------------------------------------------------------------------
-- function: return size of dataset
function DataLoader:size()
  return math.ceil(self.__size / self.batch)
end

--------------------------------------------------------------------------------
-- function: run
function DataLoader:run()
  local size, batch = self.__size, self.batch
  local idx, sample = 1, nil
  local n = 0

  local function customloop()
    if idx > size then return nil end
    local inputs, labels = self.ds:get()
    idx = idx + 1
    collectgarbage()

    sample = {inputs = inputs, labels = labels, head = head}
    n = n + 1
    return n, sample
  end

  return customloop
end

return M.DataLoader
