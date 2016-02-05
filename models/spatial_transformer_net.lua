require 'torch'
require 'nn'
require 'image'
require 'cudnn'
require 'stn'

torch.setdefaulttensortype('torch.FloatTensor')

-- Computes the number of output elements of conv layers given the input size
-- Also returns the height/width of the image
function getOutputNum(convs, input_size)
    local num_input_channels = convs[1]:get(1).nInputPlane
    local output = torch.CudaTensor(1, num_input_channels, input_size, input_size)
    for i, conv in ipairs(convs) do
        output = conv:forward(output)
    end
    return output:nElement(), output:size(3)
end

-- Creates a spatial transformer module
-- locnet are the parameters to create the localization network
-- rot, sca, tra can be used to force specific transformations
-- input_size is the height (=width) of the input
-- input_channels is the number of channels in the input
function spatialTransformer(locnet, rot, sca, tra, input_size, input_channels)
    -- Get number of params and initial state
    local init_bias = {}
    local num_params = 0
    if rot then
        num_params = num_params + 1
        init_bias[num_params] = 0
    end
    if sca then
        num_params = num_params + 1
        init_bias[num_params] = 1
    end
    if tra then
        num_params = num_params + 2
        init_bias[num_params - 1] = 0
        init_bias[num_params] = 0
    end
    if num_params == 0 then
        -- fully parameterized case
        num_params = 6
        init_bias = {1, 0, 0, 0, 1, 0}
    end

    -- Create a localization network with downsampled inputs
    local localization_network = nn.Sequential()

    local conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(input_channels, locnet[1], 5, 5, 1, 1, 2, 2))
    conv1:add(nn.SpatialBatchNormalization(locnet[1]))
    conv1:add(nn.LeakyReLU(0.1, true))
    conv1:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    conv1:cuda()

    local conv2 = nn.Sequential()
    conv2:add(cudnn.SpatialConvolution(locnet[1], locnet[2], 5, 5, 1, 1, 2, 2))
    conv2:add(nn.SpatialBatchNormalization(locnet[2]))
    conv2:add(nn.LeakyReLU(0.1, true))
    conv2:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    conv2:cuda()

    local conv_output_num = getOutputNum({conv1, conv2}, input_size / 2)

    local fc1 = nn.Sequential()
    fc1:add(nn.View(conv_output_num))
    fc1:add(nn.Linear(conv_output_num, locnet[3]))
    fc1:add(nn.LeakyReLU(0.1, true))

    local fc2 = nn.Sequential()
    fc2:add(nn.Linear(locnet[3], locnet[4]))
    fc2:add(nn.LeakyReLU(0.1, true))

    local classifier = nn.Sequential()
    classifier:add(nn.Linear(locnet[4], num_params))
    classifier:get(1).weight:zero()
    classifier:get(1).bias = torch.Tensor(init_bias)

    localization_network:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
    localization_network:add(conv1)
    localization_network:add(conv2)
    localization_network:add(fc1)
    localization_network:add(fc2)
    localization_network:add(classifier)

    -- Create the actual module structure
    local ct = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({2, 3}, {3, 4}))
    -- see (1) below
    branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))

    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    -- see (1) below
    branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))

    ct:add(branch1)
    ct:add(branch2)

    local st = nn.Sequential()

    st:add(ct)
    local sampler = nn.BilinearSamplerBHWD()

    -- (1)
    -- The sampler lead to non-reproducible results on GPU
    -- We want to always keep it on CPU
    -- This does no lead to slowdown of the training
    sampler:type('torch.FloatTensor')
    -- make sure it will not go back to the GPU when we call
    -- ":cuda()" on the network later
    sampler.type = function(type)
        return self
    end
    st:add(sampler)
    st:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor', true, true))
    st:add(nn.Transpose({3, 4}, {2, 3}))

    return st
end

function createModel()

    model = nn.Sequential()

    -- Create VGG style network first
    local vgg = nn.Sequential()

    conv32 = nn.Sequential()
    conv32:add(cudnn.SpatialConvolution(3, 32, 7, 7, 2, 2, 2, 2))
    conv32:add(nn.SpatialBatchNormalization(32))
    conv32:add(nn.LeakyReLU(0.1, true))
    conv32:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
    conv32:cuda()

    conv64 = nn.Sequential()
    conv64:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
    conv64:add(nn.SpatialBatchNormalization(64))
    conv64:add(nn.LeakyReLU(0.1, true))
    conv64:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    conv64:add(nn.SpatialBatchNormalization(64))
    conv64:add(nn.LeakyReLU(0.1, true))
    conv64:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
    conv64:cuda()

    conv128 = nn.Sequential()
    conv128:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    conv128:add(nn.SpatialBatchNormalization(128))
    conv128:add(nn.LeakyReLU(0.1, true))
    conv128:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    conv128:add(nn.SpatialBatchNormalization(128))
    conv128:add(nn.LeakyReLU(0.1, true))
    conv128:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
    conv128:cuda()

    conv256 = nn.Sequential()
    conv256:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    conv256:add(nn.SpatialBatchNormalization(256))
    conv256:add(nn.LeakyReLU(0.1, true))
    conv256:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    conv256:add(nn.SpatialBatchNormalization(256))
    conv256:add(nn.LeakyReLU(0.1, true))
    conv256:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    conv256:add(nn.SpatialBatchNormalization(256))
    conv256:add(nn.LeakyReLU(0.1, true))
    conv256:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    conv256:add(nn.SpatialBatchNormalization(256))
    conv256:add(nn.LeakyReLU(0.1, true))
    conv256:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
    conv256:cuda()

    conv512 = nn.Sequential()
    conv512:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    conv512:add(nn.SpatialBatchNormalization(512))
    conv512:add(nn.LeakyReLU(0.1, true))
    conv512:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    conv512:add(nn.SpatialBatchNormalization(512))
    conv512:add(nn.LeakyReLU(0.1, true))
    conv512:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    conv512:add(nn.SpatialBatchNormalization(512))
    conv512:add(nn.LeakyReLU(0.1, true))
    conv512:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    conv512:add(nn.SpatialBatchNormalization(512))
    conv512:add(nn.LeakyReLU(0.1, true))
    conv512:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
    conv512:cuda()

    local output_num = getOutputNum({conv32, conv64, conv128, conv256, conv512},
                                    opt.sampleSize)

    local classifier = nn.Sequential()
    classifier:add(nn.View(output_num))
    classifier:add(nn.Linear(output_num, 4096))
    classifier:add(nn.BatchNormalization(4096))
    classifier:add(nn.LeakyReLU(0.1, true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.BatchNormalization(4096))
    classifier:add(nn.LeakyReLU(0.1, true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, nClasses))
    classifier:add(nn.LogSoftMax())

    vgg:add(conv32)
    vgg:add(conv64)
    vgg:add(conv128)
    vgg:add(conv256)
    vgg:add(conv512)
    vgg:add(classifier)

    local st = spatialTransformer({64, 64, 64, 64},
                                  true, true, true,
                                  opt.sampleSize, 3)
    model:add(st)
    model:add(vgg)

    return model
end

















