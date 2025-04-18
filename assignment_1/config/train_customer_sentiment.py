# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out_scratch'
eval_interval = 300  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50  # don't print too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = 'customer-sentiment'
wandb_run_name = 'nano-gpt'

dataset = 'sentiment'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 640  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.5

learning_rate = 3e-6  # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 3e-7  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 200  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
