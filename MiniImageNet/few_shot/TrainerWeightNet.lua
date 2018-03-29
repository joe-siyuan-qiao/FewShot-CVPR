local optim = require 'optim'
paths.dofile('trainMeters.lua')

local Trainer = torch.class('Trainer')

--------------------------------------------------------------------------------
-- function: init
function Trainer:__init(model, criterion, config)
  -- training params
  self.config = config
  self.model = model
  self.net = model.net
  self.linear = model.linear
  self.criterion = nn.CrossEntropyCriterion()
  self.lr = config.lr
  self.optimState = {
    learningRate = config.lr,
    learningRateDecay = 0,
    momentum = config.momentum,
    dampening = 0,
    weightDecay = config.wd,
  }

  -- params and gradparams
  self.np, self.ng = self.net:getParameters()
  self.lp, self.lg = self.linear:getParameters()

  -- allocate cuda tensors
  self.inputs, self.labels = torch.FloatTensor(), torch.LongTensor()

  -- meters
  self.lossmeter = LossMeter()

  -- log
  self.modelsv = {model=model:clone('weight', 'bias', 'running_mean',
    'running_var'), config=config}
  self.rundir = config.rundir
  self.log = torch.DiskFile(self.rundir .. '/log', 'rw'); self.log:seekEnd()
end

--------------------------------------------------------------------------------
-- function: train
function Trainer:train(epoch, dataloader)
  self.model:training()
  self:updateScheduler(epoch)
  self.lossmeter:reset()
  local timer = torch.Timer()
  local feval = function() return self.net.output, self.ng end
  local acct, accn = 0.0, 0.0

  for n, sample in dataloader:run() do
    -- self:copySamples(sample)
    -- local outputs = self.net:forward(self.inputs)
    -- local lossbatch = self.criterion:forward(outputs, self.labels)
    -- self.net:zeroGradParameters()
    -- local gradOutputs = self.criterion:backward(outputs, self.labels)
    -- gradOutputs:mul(self.inputs:size(1))
    -- self.net:backward(self.inputs, gradOutputs)
    --
    -- -- optimize
    -- optim.sgd(feval, self.p, self.optimState)

    self:copySamples(sample)
    local weights = self.net:forward(self.inputs:narrow(1,1,80))
    self.linear.modules[1].weight:copy(weights)
    local outputs = self.linear:forward(self.inputs:narrow(1,81,80))
    local lossbatch = self.criterion:forward(outputs, self.labels)
    self.linear:zeroGradParameters()
    local gradOutputs = self.criterion:backward(outputs, self.labels)
    self.linear:backward(self.inputs:narrow(1,81,80), gradOutputs)
    self.net:zeroGradParameters()
    self.net:backward(self.inputs:narrow(1,1,80), self.lg:view(80, 64))

    local acc = 0.0
    for i = 1, 80 do
      local output = outputs[i]
      local max, ind = output:max(1)
      if i == ind[1] then acc = acc + 1 end
    end
    acct = acct + acc / 80; accn = accn + 1

    -- update loss
    self.lossmeter:add(lossbatch)

    -- optimize
    optim.sgd(feval, self.np, self.optimState)

  end

  -- write log
  local logepoch =
    string.format('[train] | epoch %05d | s/batch %04.2f | loss: %07.5f | acc: %.5f | lr: %.6f ',
      epoch, timer:time().real/dataloader:size(), self.lossmeter:value(), acct / accn, self.optimState.learningRate)
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  --save model
  torch.save(string.format('%s/model.t7', self.rundir),self.modelsv)
  if epoch%50 == 0 then
    torch.save(string.format('%s/model_%d.t7', self.rundir, epoch),
      self.modelsv)
  end

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: test
function Trainer:test(epoch, dataloader)
  self.model:evaluate()
  self.lossmeter:reset()

  for n, sample in dataloader:run() do
    self:copySamples(sample)
    local outputs = self.net:forward(self.inputs)
    local lossbatch = self.criterion:forward(outputs, self.labels)
    cutorch.synchronize()
    -- update loss
    self.lossmeter:add(lossbatch)
  end

  -- write log
  local logepoch =
    string.format('[test]  | epoch %05d | loss: %07.5f ',
      epoch, math.sqrt(self.lossmeter:value() * 50))
  print(logepoch)
  self.log:writeString(string.format('%s\n',logepoch))
  self.log:synchronize()

  collectgarbage()
end

--------------------------------------------------------------------------------
-- function: copy inputs/labels to CUDA tensor
function Trainer:copySamples(sample)
  self.inputs:resize(sample.inputs:size()):copy(sample.inputs)
  self.labels:resize(sample.labels:size()):copy(sample.labels)
end

--------------------------------------------------------------------------------
-- function: update training schedule according to epoch
function Trainer:updateScheduler(epoch)
  if self.lr == 0 then
    local regimes = {
      {   1,  50, 1e-3, 5e-4},
      {  51, 120, 5e-4, 5e-4},
      { 121, 175, 1e-4, 5e-4},
      { 176, 1e8, 1e-5, 5e-4}
    }

    for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
        -- for k,v in pairs(self.optimState) do
        --   v.learningRate=row[3]; v.weightDecay=row[4]
        -- end
        self.optimState.learningRate = row[3]
        self.optimState.weightDecay = row[4]
      end
    end
  end

  local f = io.open("lr.config", "r")
  if f then
    local lr = f:read("*number")
    if lr and lr > 0 then self.optimState.learningRate = lr end
    f:close()
  end
end

return Trainer
