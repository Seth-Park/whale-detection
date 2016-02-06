require 'csvigo'
local tds = require 'tds'

-- Read in the csv file and group the image paths by label
local f = csvigo.load({path='head_bbox.csv', mode='large'})

train = {}
val = {}
per = 0.1 -- percentage
torch.manualSeed(123)
val_size = math.ceil((#f - 1) * per)
train_size = #f - 1 - val_size
idx = torch.randperm(#f)
print(string.format("Size of total data: %d", #f - 1))
print(string.format("Size of training data: %d", train_size))
print(string.format("Size of validation data: %d", val_size))

local tr = assert(io.open('train_bbox.txt', 'w'))
for i=1, train_size do
    local index = idx[i]
    if index ~= 1 then
        local img, id, x, y, w, h = unpack(f[index])
        tr:write(img .. ',' ..
                 id .. ',' ..
                 x .. ',' ..
                 y .. ',' ..
                 w .. ',' ..
                 h .. '\n')
    end
end

local vl = assert(io.open('val_bbox.txt', 'w'))
for i=train_size+1, idx:size(1) do
    local index = idx[i]
    if index ~= 1 then
        local img, id, x, y, w, h = unpack(f[index])
        vl:write(img .. ',' ..
                 id .. ',' ..
                 x .. ',' ..
                 y .. ',' ..
                 w .. ',' ..
                 h .. '\n')
    end
end
tr:close()
vl:close()

print('Train/Validation split completed')



