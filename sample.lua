
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

local cunn_loaded, cunn = pcall(require, 'cunn')
local cutorch_loaded, cutorch = pcall(require, 'cutorch')
local clnn_loaded, clnn = pcall(require, 'clnn')
local cltorch_loaded, cltorch = pcall(require, 'cltorch')

require 'util.GloVeEmbedding'
require 'util.OneHot'

local CharSplitLMMinibatchLoader 	= require 'util.CharSplitLMMinibatchLoader'
local WordSplitLMMinibatchLoader 	= require 'util.WordSplitLMMinibatchLoader'
local config						= require 'config'

-- gated print: simple utility function wrapping a print
function gprint(str)
    if config.verbose == 1 then print(str) end
end

-- Set Checkpoint File
if string.len(config.checkpoint) > 0 then
	config.checkpoint_file = config.checkpoint_dir .. '/' .. config.checkpoint
else
	config.checkpoint_file = string.format('%s/lm_%s_epoch%.2f.t7', config.checkpoint_dir, config.savefile, config.max_epochs)
end

-- load the model checkpoint
if not lfs.attributes(config.checkpoint_file, 'mode') then
    gprint('Error: File ' .. config.checkpoint_file .. ' does not exist.')
else
	gprint('Loading Checkpoint File: ' .. config.checkpoint_file )
end

-- Load Checkpoint
checkpoint 	= torch.load(config.checkpoint_file)

-- Set Checkpoint Options
config.seed	= checkpoint.opt.seed
config.backend = checkpoint.opt.backend
config.gpuid = checkpoint.opt.gpuid


-- check that cunn/cutorch are installed if user wants to use the GPU
if config.gpuid >= 0 and config.backend == 'cuda' then
    if not cunn_loaded then gprint('package cunn not found!') end
    if not cutorch_loaded then gprint('package cutorch not found!') end
    if cunn_loaded and cutorch_loaded then
        gprint('using CUDA on GPU ' .. config.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(config.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(config.seed)
    else
        gprint('Falling back on CPU mode')
        config.backend = 'cpu' -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if config.gpuid >= 0 and config.backend == 'cl' then

    if not clnn_loaded then print('package clnn not found!') end
    if not cltorch_loaded then print('package cltorch not found!') end
    if clnn_loaded and cltorch_loaded then
        gprint('using OpenCL on GPU ' .. config.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(config.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(config.seed)
    else
        gprint('Falling back on CPU mode')
        config.backend = 'cpu' -- overwrite user setting
    end
end

torch.manualSeed(config.seed)

protos = checkpoint.protos

protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros

gprint('creating an ' .. checkpoint.opt.model .. '...')

local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if config.gpuid >= 0 and config.backend == 'cuda' then h_init = h_init:cuda() end
    if config.gpuid >= 0 and config.backend == 'cl' then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state

-- Specify file
assert(string.len(config.output_file) > 0, 'Error, An output file name must be specified in the config.lua file')
local file = io.open(config.output_file, "w")

-- do a few seeded timesteps
local seed_text = config.primetext
local unknownword = "<unk>"


	if string.len(seed_text) > 0 then
	    gprint('seeding with ' .. seed_text)
	    gprint('--------------------------')
	    
	    
	    local seedlist
	    if(checkpoint.opt.wordlevel==1) then
	       local words=WordSplitLMMinibatchLoader.preprocess(seed_text)
	       seedlist = words:gmatch("([^%s]+)")
	    else
	        seedlist = seed_text:gmatch'.'
	    
	    end
	    
	    
	    for c in seedlist do
	    
	        local idx = vocab[c]

	        if idx == nil then 
	        	idx = vocab[unknownword]
	        	gprint(c .. ' does not exist in vocabulary. Vocabulary size: ' .. #vocab)
			    -- fill with uniform probabilities over characters (? hmm)
			    gprint('missing seed text, using uniform probability over first character')
			    gprint('--------------------------')
			    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
			    if config.gpuid >= 0 and config.backend == 'cuda' then prediction = prediction:cuda() end
			    if config.gpuid >= 0 and config.backend == 'cl' then prediction = prediction:cl() end
	        else
		        prev_char = torch.Tensor{vocab[c]}
		        file:write( ivocab[prev_char[1]] .. " " )
		        if config.gpuid >= 0 and config.backend == 'cuda' then prev_char = prev_char:cuda() end
		        if config.gpuid >= 0 and config.backend == 'cl' then prev_char = prev_char:cl() end
		        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
		        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
		        current_state = {}
		        for i=1,state_size do table.insert(current_state, lst[i]) end
		        prediction = lst[#lst] -- last element holds the log probabilities
	        end
	    end
	else
	
	    -- fill with uniform probabilities over characters (? hmm)
	    gprint('missing seed text, using uniform probability over first character')
	    gprint('--------------------------')
	    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
	    if config.gpuid >= 0 and config.backend == 'cuda' then prediction = prediction:cuda() end
	    if config.gpuid >= 0 and config.backend == 'cl' then prediction = prediction:cl() end
	    
	end

-- start sampling/argmaxing
for i=1, config.length do

    -- log probabilities from the previous timestep
    if config.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        prediction:div(config.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        if config.skip_unk then
            prev_char = torch.multinomial(probs:float(), 2):float()
            prev_char = prev_char[1] == vocab["<unk>"] and prev_char[{{2}}] or prev_char[{{1}}]
        else
            prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
        end

    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probabilities


    local word = ivocab[prev_char[1]]
    if(checkpoint.opt.wordlevel==1) then
      word=word.." "
    end
    file:write(word)
end
file:write('\n') file:flush()