local opt = {}

-- Chord-RNN Configuration


-- Train Configuration

-- Data Directory
opt.data_dir					= 'data/firstpage'			--data directory. Should contain the file input.txt with input data

-- model params
opt.rnn_size 					= 512 			--size of LSTM internal state
opt.num_layers 					= 2				--number of layers in the LSTM
opt.model						= 'lstm'		--(lstm,gru or rnn)
opt.wordlevel 					= 1				--(1 for word level 0 for char)

-- optimization
opt.learning_rate 				= 3e-3			--learning rate
opt.learning_rate_decay 		= 0.97			--learning rate decay
opt.learning_rate_decay_after 	= 5				--in number of epochs, when to start decaying the learning rate
opt.decay_rate 					= 0.95			--decay rate for rmsprop
opt.dropout 					= .25			--dropout for regularization, used after each RNN hidden layer. (0 = no dropout)
opt.seq_length 					= 80			--number of timesteps to unroll for
opt.batch_size 					= 10			--number of sequences to train on in parallel
opt.max_epochs 					= 100			--number of full passes through the training data
opt.grad_clip 					= 3				--clip gradients at this value
opt.train_frac 					= 0.80			--fraction of data that goes into train set
opt.val_frac 					= 0.20			--fraction of data that goes into validation set
            									--test_frac will be computed as (1 - train_frac - val_frac)
opt.init_from 					= ''			--initialize network parameters from checkpoint at this path
opt.optim	 					= 'rmsprop' 	--which optimizer to use: (rmsprop|sgd|adagrad|asgd|adam)
opt.optim_alpha					= 0.8			--alpha for adagrad/rmsprop/momentum/adam
opt.optim_beta					= 0.999			--beta used for adam
opt.optim_epsilon				= 1e-8			--epsilon that goes into denominator for smoothing


-- bookkeeping
opt.seed 						= 123			--torch manual random number generator seed
opt.print_every 				= 1				--how many steps/minibatches between printing out the loss
opt.eval_val_every 				= 200			--every how many iterations should we evaluate on validation data?
opt.checkpoint_dir 				= 'cv' 			--output directory where checkpoints get written
opt.savefile 					= 'checkpoint' 	--filename to autosave the checkpont to. Will be inside cv/
opt.threshold 					= 10			--minimum number of occurences a token must have to be included 
												--(ignored if -wordlevel is 0)

-- GPU/CPU
opt.backend 					= 'cl'			--(cpu|cuda|cl)
opt.gpuid						= 0				--which gpu to use (ignored if backend is cpu)

-- Glove
opt.glove 						= 1									--whether or not to use GloVe embeddings
opt.embedding_file 				= 'util/glove/glove.840B.300d.txt'	--filename of the glove (or other) embedding file
opt.embedding_file_size 		= 300								--feature vector size of embedding file




-- Sampling Configuration


-- checkpoint
opt.checkpoint					= ''								--model checkpoint to use for sampling.  If Empty, pulls last checkpoint

-- Sampling
opt.sample						= 1									--(0 to use max at each timestep, 1 to sample at each timestep)
opt.primetext					= 'the'								--used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample
opt.length						= 500								--number of characters to sample
opt.temperature					= .2								--temperature of sampling
opt.skip_unk					= 1									--whether to skip UNK tokens when sampling

-- Bookkeeping
opt.verbose						= 1									--set to 0 for no console diagnostics
opt.output_file					= 'output.txt'						--specify the name of the saved output file


return opt
