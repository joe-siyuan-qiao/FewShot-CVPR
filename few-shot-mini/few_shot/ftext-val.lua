require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'image'

local t = require 'transforms'

--------------------------------------------------------------------------------
---- parse arguments
local cmd = torch.CmdLine()
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-dim', 64, 'dimension')

local config = cmd:parse(arg)

--------------------------------------------------------------------------------
---- initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

--------------------------------------------------------------------------------
---- start program
local modelDir = paths.concat('pretrained', 'model.t7')
local dataDir = paths.concat('../data', 'test')
local interDir = paths.concat('intermediate')
local meanStd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}
local preprocess = t.Compose{
    t.Scale(92),
    t.ColorNormalize(meanStd),
    t.CenterCrop(80),
}

local net = torch.load(modelDir)
local linear = net.modules[#net.modules]:clone()
net:remove()
net:cuda(); linear:cuda()
net:evaluate()
local inpPad = torch.CudaTensor()
local classTable = {}
for file in lfs.dir(dataDir) do
  if lfs.attributes(paths.concat(dataDir, file), "mode") == "directory" and
  file ~= "." and file ~= ".." then
    table.insert(classTable, file)
  end
end
local input, label = {}, torch.FloatTensor()
input.indv = {}
input.mean = {}
local weight = linear.weight:float()
local bias = linear.bias:float()
label:resize(weight:size(1), weight:size(2) + 1)
label:narrow(2, 1, weight:size(2)):copy(weight)
label:narrow(2, weight:size(2) + 1, 1):copy(bias)

for cid, class in pairs(classTable) do
  local classImgTable = {}
  for file in lfs.dir(paths.concat(dataDir, class)) do
    if file ~= "." and file ~= ".." then
      table.insert(classImgTable, file)
    end
  end
  outAcc = torch.FloatTensor(config.dim):fill(0)
  input.indv[class] = {}
  input.mean[class] = {}
  for iid, img in pairs(classImgTable) do
    io.write(string.format('\r| ftext | %05d / %05d | %05d / %05d |',
        cid, #classTable, iid, #classImgTable))
    io.flush()
    local imgPath = paths.concat(dataDir, class, img)
    local img = image.load(imgPath, 3)
    img = preprocess(img)
    inpPad:resize(1,3,80,80):fill(.5)
    inpPad:narrow(1,1,1):narrow(3,1,80):narrow(4,1,80):copy(img)
    cutorch.synchronize()
    local out = net:forward(inpPad)
    local pred = linear:forward(out)
    cutorch.synchronize()
    out = out:float():view(config.dim)
    pred = pred:float():view(80)
    local max, ind = torch.max(pred, 1)
    table.insert(input.indv[class], out:clone())
    outAcc:add(out)
  end
  outAcc:div(#classImgTable)
  local out = linear:forward(outAcc:cuda())
  cutorch.synchronize()
  out = out:float():view(80)
  local max, ind = torch.max(out, 1)
  table.insert(input.mean[class], {outAcc:clone(), ind[1]})
  collectgarbage()
end
print (' done')
torch.save('featurepool/ft-val.t7', input.indv)
