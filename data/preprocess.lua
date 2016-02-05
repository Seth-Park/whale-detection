require 'csvigo'
local tds = require 'tds'

-- Read in the csv file and group the image paths by label
local f = csvigo.load({path='trainLabels.csv', mode='large'})

id_class = {} -- mapping between whale ID and class number
class_num = 1
label_img_pair = {}

for i=2, #f do
    local img, id = unpack(f[i])
    if not id_class[id] then
        id_class[id] = class_num
        label_img_pair[class_num] = {}
        class_num = class_num + 1
    end
    local label = id_class[id]
    label_img_pair[label][#label_img_pair[label] + 1] = img
end

-- Split the data into training set and validation set with the corresponding labels
-- For labels that only have one image, simply include it in the training set
per = 0.1 -- percentage
train = {}
val = {}
math.randomseed(10)

for i=1, #label_img_pair do
    local images = label_img_pair[i]
    if #images ~= 1 then
        local nval = math.ceil(per * #images)
        local val_imgs = {}

        for i=1, nval do
            local idx = math.random(1, #images)
            while val_imgs[idx] do idx = math.random(1, #images) end
            val_imgs[idx] = true
        end

        for j=1, #images do
            if val_imgs[j] then
                val[#val + 1] = {images[j], i}
            else
                train[#train + 1] = {images[j], i}
            end
        end
    else
        train[#train + 1] = {images[1], i}
    end
end

print(string.format("Size of total data: %d", #f - 1))
print(string.format("Number of class: %d", #label_img_pair))
print(string.format("Size of training data: %d", #train))
print(string.format("Size of validation data: %d", #val))

-- Write the (image, label) pair into .txt files
local tr = assert(io.open('train_labels.txt', 'w'))
local vl = assert(io.open('val_labels.txt', 'w'))

for i=1, #train do
    local image, label = unpack(train[i])
    tr:write(image .. ',' .. label .. '\n')
end

for i=1, #val do
    local image, label = unpack(val[i])
    vl:write(image .. ',' .. label .. '\n')
end

tr:close()
vl:close()
print('Train/Validation split completed')



