-- Referenced from github.com/soumith

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
nTest = nil
nClasses = 427

if opt.nDonkeys > 0 then
       local options = opt -- make an upvalue to serialize over to donkey threads
          donkeys = Threads(opt.nDonkeys,
                            function()
                                require 'torch';
                                local Threads = require 'threads';
                                Threads.serialization('threads.sharedserialize')
                            end,
                            function(idx)
                                opt = options
                                tid = idx
                                local seed = opt.manualSeed + idx
                                torch.manualSeed(seed)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                paths.dofile('donkey.lua')
                            end
                    );
else -- singled threaded data loading. Useful for debugging
    paths.dofile('donkey.lua')
    donkeys = {}
    function donkeys:addjob(f1, f2) f2(f1()) end
    function donkeys:synchronize() end
end

donkeys:addjob(function() return #val_labels end, function(v) nTest = v end)
donkeys:synchronize()
assert(nTest)

