-- the script to evalute the n-shot k-way task
require 'nn'
require 'WeightNet'
torch.manualSeed(0)

local cmd = torch.CmdLine()
cmd:option('-weightnet', 'pretrained/weightnet.t7', '')
config = cmd:parse(arg)

local fttab = torch.load('featurepool/ft-val.t7')
local wnmod = torch.load(config.weightnet)
wnmod = wnmod.model.net
wnmod:evaluate()
local class = {}
for cls, _ in pairs(fttab) do
    table.insert(class, cls)
end

-- find 5 classes along with their images
function find5Class()
    local tab = {}
    local idx = torch.randperm(#class)
    idx = idx:narrow(1, 1, 5)
    for i = 1, 5 do tab[i] = fttab[class[idx[i]]] end
    return tab
end

-- given classes with images, split them into test and reference
-- it's assumed that the first is the reference
function split(tab)
    for i = 1, 5 do
        for j = 1, 5 do
            local ftvec = tab[i]
            local rdidx = torch.random(#ftvec)
            local fttmp = ftvec[rdidx]
            ftvec[rdidx] = ftvec[j]
            ftvec[j] = fttmp
        end
    end
end

-- preprocess function
function p(a)
    return wnmod:forward(a):clone()
end

-- start test
-- one-shot
local acc_all = {}
for _ = 1, 1000 do
    local tab = find5Class()
    split(tab)
    local ref = {}
    for i = 1, 5 do ref[i] = p(tab[i][1]) end
    local right, wrong = 0, 0
    for i = 1, 5 do
        for j = 2, #tab[i] do
            local maxSim, maxIdx = -1e12, 0
            for k = 1, 5 do
                local sim = torch.dot(ref[k], tab[i][j])
                if sim > maxSim then maxSim = sim; maxIdx = k end
            end
            if maxIdx == i then
                right = right + 1
            else
                wrong = wrong + 1
            end
        end
    end
    table.insert(acc_all, right / (right + wrong))
end
acc_all = torch.Tensor(acc_all)
print (acc_all:mean() .. ' ' .. acc_all:std())

-- 5-shot (mean)
local acc_all = {}
for _ = 1, 1000 do
    local tab = find5Class()
    split(tab)
    local ref = {}
    for i = 1, 5 do ref[i] = tab[i][1]:clone() end
    for i = 1, 5 do for j = 2, 5 do ref[i]:add(tab[i][j]) end end
    for i = 1, 5 do ref[i]:div(5) end
    for i = 1, 5 do ref[i] = p(ref[i]) end
    local right, wrong = 0, 0
    for i = 1, 5 do
        for j = 5, #tab[i] do
            local maxSim, maxIdx = -1e12, 0
            for k = 1, 5 do
                local sim = torch.dot(ref[k], tab[i][j])
                if sim > maxSim then maxSim = sim; maxIdx = k end
            end
            if maxIdx == i then
                right = right + 1
            else
                wrong = wrong + 1
            end
        end
    end
    table.insert(acc_all, right / (right + wrong))
end
acc_all = torch.Tensor(acc_all)
print (acc_all:mean() .. ' ' .. acc_all:std())
