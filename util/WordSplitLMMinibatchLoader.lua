
--Modified by Mike Tanana from Andrew Karpathy and Wojciech Zaremba 
--Changed to support word models

local WordSplitLMMinibatchLoader = {}
WordSplitLMMinibatchLoader.__index = WordSplitLMMinibatchLoader

function WordSplitLMMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, threshold)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, WordSplitLMMinibatchLoader)

    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocabwords.t7')
    local tensor_file = path.join(data_dir, 'datawords.t7')

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        -- prepro files do not exist, generate them
        print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(input_file)
        local vocab_attr = lfs.attributes(vocab_file)
        local tensor_attr = lfs.attributes(tensor_file)
        if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
            print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ' .. input_file .. '...')
        WordSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, threshold)
    end

    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end
    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    
    -- Y is the next letter used for the prediction.
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then 
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = self.nbatches - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.nbatches * split_fractions[1])
        self.nval = math.floor(self.nbatches * split_fractions[2])
        self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    print(string.format('Vocab Size: %d, Threshold: %d', self.vocab_size, threshold))
		    
    collectgarbage()
    return self
end

function WordSplitLMMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end



function WordSplitLMMinibatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_batches[ix], self.y_batches[ix]
end


function WordSplitLMMinibatchLoader.preprocess(alltext)
  --make sure there are spaces around certain characters so that we predict them as individual units
  local newtext
  newtext = alltext:gsub(',',' , ')     
  newtext = newtext:gsub('%.',' . ')  
  newtext = newtext:gsub('%:',' : ')  
  newtext = newtext:gsub('%;',' ; ') 
  newtext = newtext:gsub('%?',' ? ')
  newtext = newtext:gsub('%!',' ! ') 
  newtext = newtext:gsub('\n',' \n ') 


	return newtext
end


-- *** STATIC method ***
function WordSplitLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, threshold)
    --local timer = torch.Timer()
    local matchstring = "([^%s]+)"
    print('loading text file...')
    local wordcount = {}
    local rawdata
    local tot_len = 0
    local filein = io.open(in_textfile, "r")
    local unknownword = "<unk>"
    local trimvalue = threshold

    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    print('word occurence threshold is ' .. threshold)
    -- record all characters to a set
    local unordered = {}
    local count=0
    local t = filein:read("*all")
    t=WordSplitLMMinibatchLoader.preprocess(t)
    local words = t:gmatch(matchstring )
    
   
    for word in words do
       word = word:lower()
       
          if wordcount[word]==nil then
              wordcount[word]=1
          else
               wordcount[word]=wordcount[word]+1
          end
         tot_len=tot_len+1
         
    end

    filein:close()

 
    
    local vocab_mapping = {}
    local index=1
    vocab_mapping[unknownword]=index;
    index=index+1
    for key,value in pairs(wordcount) do 
      if(value>=trimvalue) then  --trim dictionary for rare words
        vocab_mapping[key]=index;
        index=index+1
        count=count+1
      end
    end
    
     -- f:close()
    
    --later on we can decide to trim this vocab if we'd like
    --print(wordcount)
    print("Count: "..count)
    print("Length: "..tot_len)

    
     -- construct a tensor with all the data
    print('putting data into tensor...')
    local data = torch.IntTensor(tot_len) -- store it into 1D first, then rearrange
    filein = io.open(in_textfile, "r")
    t = filein:read("*all")
    t=WordSplitLMMinibatchLoader.preprocess(t)
    words = t:gmatch(matchstring )
    local loc=1
    for word in words do
       word = word:lower()
       local idx = vocab_mapping[word]
       if idx == nil then idx = vocab_mapping[unknownword] end
       data[loc] = idx
       if vocab_mapping[word]==0 then print("ZERO INDEX "..word) end 
       loc=loc+1
    end
    
 
     -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    print('Vocab Mappings Size:'..tablelength(vocab_mapping))
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
 	print('Tensor File Size:' .. data:size(1) )
    torch.save(out_tensorfile, data)
    
    
   
   
end

function tablelength(T)
  local count = 0
  for _,_ in pairs(T) do count = count + 1 end
  return count
end


return WordSplitLMMinibatchLoader

