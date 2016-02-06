require 'torch'
require 'nn'
require 'image'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

-- Computes the number of output elements of conv layers given the input size
-- Also return the height/width of the image
function getOutputInfo(convs, input_size)
    local num_input_channels = convs[1]:get(1).nInputPlane
    local output = toch.CudaTensor(1, num_input_channels, input_size, input_size)
    for i, conv in ipairs(convs) do
        output = conv:forward(output)
    end
    return output:nElement(), output:size(3)
end

-- Creates a head localizer network
function createModel()
    local input_channel = 3
    local network_dim = { 16, 64, 64, 64, 64, 1028, 1028 }
    local output_dim = 4


    local head_localizer = nn.Sequential()

    head_localizer:add(cudnn.SpatialConvolution(input_channel,
                                                network_dim[1],
                                                3, 3, 1, 1, 1, 1))
    head_localizer:add(nn.SpatialBatchNormalization(network_dim[1]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    head_localizer:add(cudnn.SpatialConvolution(network_dim[1],
                                                network_dim[2],
                                                3, 3, 1, 1, 1, 1))
    head_localizer:add(nn.SpatialBatchNormalization(network_dim[2]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    head_localizer:add(cudnn.SpatialConvolution(network_dim[2],
                                                network_dim[3],
                                                3, 3, 1, 1, 1, 1))
    head_localizer:add(nn.SpatialBatchNormalization(network_dim[3]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    head_localizer:add(cudnn.SpatialConvolution(network_dim[3],
                                                network_dim[4],
                                                3, 3, 1, 1, 1, 1))
    head_localizer:add(nn.SpatialBatchNormalization(network_dim[4]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    head_localizer:add(cudnn.SpatialConvolution(network_dim[4],
                                                network_dim[5],
                                                3, 3, 1, 1, 1, 1))
    head_localizer:add(nn.SpatialBatchNormalization(network_dim[5]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))

    output_info = getOutputInfo({head_localizer}, opt.sampleSize)

    head_localizer:add(nn.View(output_info))
    head_localizer:add(nn.Linear(output_info, network_dim[6]))
    head_localizer:add(nn.BatchNormalization(network_dim[6]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(nn.Linear(network_dim[6], network_dim[7]))
    head_localizer:add(nn.BatchNormalization(network_dim[7]))
    head_localizer:add(cudnn.ReLU(true))
    head_localizer:add(nn.Linear(network_dim[7], output_dim))

    return head_localizer


