require 'torch'
require 'cutorch'

--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('train WeightNet')
cmd:text()
cmd:text('Options:')
cmd:option('-rundir', 'exps/', 'experiments directory')
cmd:option('-datadir', 'data/Data/CLS-LOC/train', 'data directory')
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-nthreads', 4, 'number of threads for DataSampler')
cmd:option('-reload', '', 'reload a network from given directory')
cmd:option('-modeldir', 'pretrained/resnet-50.t7', 'location for resnet')
cmd:text()
cmd:text('Training Options:')
cmd:option('-fft', true, 'training using pre-computed features')
cmd:option('-ftdir', 'featurepool/ft.t7', 'location of pre-computed features')
cmd:option('-mpdir', 'featurepool/mapping.t7', 'the mapping directory')
cmd:option('-lr', 0, 'learning rate (0 uses default lr schedule)')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-wd', 5e-4, 'weight decay')
cmd:option('-maxload', 250, 'max number of training batches per epoch')
cmd:option('-testmaxload', 30, 'max number of testing batches')
cmd:option('-maxepoch', 50, 'max number of training epochs')
cmd:option('-reloadepoch', 1, 'the starting epoch for reloading')

local config = cmd:parse(arg)
local configrundir = config.rundir
local configreloadepoch = config.reloadepoch
config.batch = 900

--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)
math.randomseed(config.seed)

paths.dofile('WeightNet.lua')

--------------------------------------------------------------------------------
-- reload?
local epoch, model
local reloadpath = config.reload
if #config.reload > 0 then
  epoch = 0
  if paths.filep(config.reload..'/log') then
    for line in io.lines(config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', config.reload))
  local m = torch.load(string.format('%s/model.t7', config.reload))
  model, config = m.model, m.config
end

--------------------------------------------------------------------------------
-- directory to save log and model
local pathsv = 'weightnet/exp'
config.rundir = cmd:string(
  paths.concat(configrundir, pathsv),
  config,{rundir=true, gpu=true, reload=true, datadir=true, dm=true} --ignore
)

print(string.format('| running in directory %s', config.rundir))
os.execute(string.format('mkdir -p %s',config.rundir))

model = model or nn.WeightNet(config)
local criterion = nn.MSECriterion():cuda()

--------------------------------------------------------------------------------
-- initialize data loader
local DataLoader = paths.dofile('DataLoader.lua')
local trainLoader, valLoader = DataLoader.create(config)

--------------------------------------------------------------------------------
-- initialize Trainer (handles training/testing loop)
paths.dofile('TrainerWeightNet.lua')
local trainer = Trainer(model, criterion, config)

--------------------------------------------------------------------------------
-- do it
epoch = configreloadepoch
print('| start training')
for i = 1, config.maxepoch do
  trainer:train(epoch,trainLoader)
  epoch = epoch + 1
end
