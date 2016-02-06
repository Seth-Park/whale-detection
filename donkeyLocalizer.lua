-- Load images, preprocess and augment them, and create a batch
-- Referenced from github.com/soumith

require 'image'
tds=require 'tds'
utils=paths.dofile('utils.lua') -- utils.lua in same directory
torch.setdefaulttensortype('torch.FloatTensor')
local trainSize = {3, opt.loadSize, opt.loadSize}
local imagesRoot = paths.concat(opt.dataRoot, 'train_' .. opt.loadSize)

local function loadImage(rawJPG)
    local input = image.decompressJPG(rawJPG, 3, 'float')
    local iH = input:size(2)
    local iW = input:size(3)
    -- resize
    input = image.scale(input, opt.loadSize, opt.loadSize)
    -- normalize
    for i=1,3 do -- channels
        if mean then input[{{i},{},{}}]:add(-mean[i]) end
        if std then input[{{i},{},{}}]:div(std[i]) end
    end
    return input, iH, iW
end

local function processTrain(rawJPG)
    collectgarbage()
    local input, input_h, input_w = loadImage(rawJPG)
    out = image.rotate(out, torch.uniform() * math.pi * 0.06, 'bilinear')    -- rotation jitter
    return out, input_h, input_w
end

local function processTest(rawJPG)
    collectgarbage()
    local out, out_h, out_w = loadImage(rawJPG)
    return out, out_h, out_w
end

function getTrainingMiniBatch(quantity)
    local data = torch.Tensor(quantity, trainSize[1], trainSize[2], trainSize[3])
    local label = torch.Tensor(quantity, 4)
    for i=1, quantity do
        local index = torch.random(1, #train_data)
        local out, input_h, input_w = processTrain(train_data[index])
        data[i]:copy(out)
        x, y, w, h = train_labels[i]
        w_scale = opt.loadSize / input_w
        h_scale = opt.loadSize / input_h
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        label[i] = torch.Tensor({x, y, w, h})
    end
    return data, label
end

function getValidationData(i1, i2)
    local data = torch.Tensor(i2-i1+1, trainSize[1], trainSize[2], trainSize[3])
    local label = torch.Tensor(i2-i1+1, 4)
    for i=i1, i2 do
        local out, out_h, out_w = processTest(val_paths[i])
        x, y, w, h = val_labels[i]
        w_scale = opt.loadSize / out_w
        h_scale = opt.loadSize / out_h
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        data[i-i1+1]:copy(out)
        label[i-i1+1] = torch.Tensor({x, y, w, h})
    end
    return data, label
end

-------------------------
-- Load training data
-------------------------
-- train data is stored in a simple way using tds.hash
train_data = tds.hash()
train_labels = tds.hash()
-- load labels from file
for l in io.lines(paths.concat(opt.dataRoot, 'train_labels.txt')) do
    local path, whaleID, x, y, w, h = unpack(l:split(','))
    x = math.floor(x)
    y = math.floor(y)
    w = math.floor(w)
    h = math.floor(h)
    train_data[#train_data + 1]
    = utils.loadFileAsByteTensor(paths.concat(imagesRoot, path))
    train_labels[#train_labels + 1] = {x, y, w, h}
end
-- val data is stored in the same way
val_paths = tds.hash()
val_labels = tds.hash()
for l in io.lines(paths.concat(opt.dataRoot, 'val_labels.txt')) do
    local path, whaleID, x, y, w, h = unpack(l:split(','))
    x = math.floor(x)
    y = math.floor(y)
    w = math.floor(w)
    h = math.floor(h)
    val_paths[#val_paths+1]
    = utils.loadFileAsByteTensor(paths.concat(imagesRoot, path))
    val_labels[#val_labels+1] = {x, y, w, h}
end

collectgarbage()
-----------------------------------------
-- estimate mean/std per channel
-----------------------------------------
do
    local meanstdCacheFile = 'meanstdCache_256.t7'
    if paths.filep(meanstdCacheFile) then
        print('Loading mean/std from cache file')
        local meanstd = torch.load(meanstdCacheFile)
        mean = meanstd.mean
        std  = meanstd.std
    else
        print('Estimating mean/std from a few images in dataset. Will be cached for future use.')
        local nSamples = 1000
        local meanEstimate = {0,0,0}
        for i=1,nSamples do
            local img = getTrainingMiniBatch(1)[1]
            for j=1,3 do meanEstimate[j] = meanEstimate[j] + img[j]:mean() end
        end
        for j=1,3 do meanEstimate[j] = meanEstimate[j] / nSamples end
        mean = meanEstimate
        local stdEstimate = {0,0,0}
        for i=1,nSamples do
            local img = getTrainingMiniBatch(1)[1]
            for j=1,3 do stdEstimate[j] = stdEstimate[j] + img[j]:std() end
        end
        for j=1,3 do stdEstimate[j] = stdEstimate[j] / nSamples end
        std = stdEstimate
        torch.save(meanstdCacheFile, {mean=mean, std=std})
    end
end
