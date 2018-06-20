require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'hdf5'
require 'rnn'
cjson=require('cjson')
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate the ACP module in GVQA')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_data_folder','processed_inputs/','path to the data folder')
cmd:option('-input_ques_h5','processed_question_for_acp.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','input_json_for_acp.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'models/', 'path to folder containing different models')
cmd:option('-model_name', 'model_acp.t7', 'name of the model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'predictions/', 'path to save output json file')
cmd:option('-input_question_classifier_output', 'predictions/question_classifier_results.h5', 'path to yes no classifier output')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-question_word_encoding','glove','glove or one-hot')
cmd:option('-batch_size',4096,'batch_size for each iterations')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',256,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',1,'number of the rnn layer')
cmd:option('-num_fc_layers',1,'number of fc layers')
cmd:option('-size_last_fc_layer',256,'size of last hidden fc layer')
cmd:option('-num_answer_clusters', 50, 'number of answer clusters for ACP module')

-- Miscellaneous
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-num_ques', -1, 'number of questions to use')

opt = cmd:parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------

print('DataLoader loading h5 file: ', opt.input_json)

local file = io.open(opt.input_data_folder .. opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading question classifier output')
local h5_file = hdf5.open(opt.input_question_classifier_output, 'r')
local yes_no_out_qids = h5_file:read('/qids'):all() 
local yes_no_out_pred = h5_file:read('/pred'):all() 
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_data_folder .. opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_test'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
dataset['ques_id'] = h5_file:read('/question_id_test'):all()
h5_file:close()

print('fetching non yes-no data')
local qids_to_idx = {}
for i = 1, dataset['ques_id']:size(1) do
    qids_to_idx[dataset['ques_id'][i]] = i
end

local des_inds = {}
local num_data_points = yes_no_out_qids:size(1)
local count = 0
for i = 1, num_data_points do
    if yes_no_out_pred[i] == 0 then
        count = count + 1
        des_inds[count] = qids_to_idx[yes_no_out_qids[i]]
    end
end

des_inds = torch.LongTensor(des_inds)

dataset['question'] = dataset['question']:index(1,des_inds)
dataset['lengths_q'] = dataset['lengths_q']:index(1,des_inds)
dataset['ques_id'] = dataset['ques_id']:index(1,des_inds)

dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

if opt.question_word_encoding == 'one-hot' then
    local count = 0
    for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
    vocabulary_size_q=count
elseif opt.question_word_encoding == 'glove' then
    ques_word_embedding_dim = 300
end

collectgarbage()

------------------------------------------------------------------------
-- Design Parameters and Network Definitions
------------------------------------------------------------------------

if opt.question_word_encoding == 'one-hot' then
    -- LookupTable + RNN
    rnn_q = nn.Sequential()
        :add(nn.LookupTableMaskZero(vocabulary_size_q, opt.input_encoding_size))
        :add(nn.Dropout(0.5))
        :add(nn.SplitTable(1, 2))
        :add(nn.Sequencer(nn.FastLSTM(opt.input_encoding_size, opt.rnn_size):maskZero(1)))
elseif opt.question_word_encoding == 'glove' then
       -- linear layer + RNN
    rnn_q = nn.Sequential()
        :add(nn.Bottle(nn.MaskZero(nn.Linear(ques_word_embedding_dim, opt.input_encoding_size), 2)))
        :add(nn.Dropout(0.5))
        :add(nn.Tanh())
        :add(nn.SplitTable(1, 2))
        :add(nn.Sequencer(nn.FastLSTM(opt.input_encoding_size, opt.rnn_size):maskZero(1)))
end

if opt.rnn_layer > 1 then
    for i = 1, opt.rnn_layer - 1 do
        rnn_q:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size):maskZero(1)))
    end
end

rnn_q:add(nn.SelectTable(-1))
     :add(nn.Dropout(0.5))

if opt.num_fc_layers == 1 then
    rnn_q:add(nn.Linear(opt.rnn_size, opt.num_answer_clusters))
elseif opt.num_fc_layers == 2 then
    rnn_q:add(nn.Linear(opt.rnn_size, opt.size_last_fc_layer))
         :add(nn.Tanh())
         :add(nn.Dropout())
         :add(nn.Linear(opt.size_last_fc_layer, opt.num_answer_clusters))
end

softmax_net_acp = nn.SoftMax()

if opt.gpuid >= 0 then
    print('shipped data function to cuda...')
    rnn_q = rnn_q:cuda()
end

-- setting to evaluation
rnn_q:evaluate()

rnn_q_w, rnn_q_dw = rnn_q:getParameters()

-- loading the model
model_param = torch.load(opt.model_path .. opt.model_name)
rnn_q_w:copy(model_param['rnn_q_w'])

sizes = {rnn_q_w:size(1)}

------------------------------------------------------------------------
-- Grab Next Batch --
------------------------------------------------------------------------

function dataset:next_batch_test(s, e)

    local batch_size = e - s + 1
    local qinds = torch.LongTensor(batch_size):fill(0)

    for i = 1, batch_size do
        qinds[i] = s+i-1
    end

    local fv_q = dataset['question']:index(1,qinds)
    local fv_q_len = dataset['question']:index(1,qinds)

    local qids = dataset['ques_id']:index(1,qinds)

    -- ship to gpu
    if opt.gpuid >= 0 then
        fv_q = fv_q:cuda()
    end

    return fv_q, qids
end

function forward(s,e)
    -- get a batch
    local fv_q, qids = dataset:next_batch_test(s, e)

    -- LookupTable + RNN forward pass
    local scores = rnn_q:forward(fv_q)

    return scores:double(), qids

end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------

nqs = dataset['question']:size(1)
if opt.num_ques >= 0 then
    nqs = opt.num_ques
end
scores = torch.Tensor(nqs, opt.num_answer_clusters)

qids = torch.LongTensor(nqs)

for i = 1, nqs, opt.batch_size do
    xlua.progress(i, nqs)
    r = math.min(i + opt.batch_size - 1, nqs)
    scores[{{i, r},{}}], qids[{{i, r}}] = forward(i, r)
end

print('saving predicted cluster_ids')
paths.mkdir(opt.out_path)
local myFile = hdf5.open(opt.out_path .. 'acp_results.h5', 'w');
myFile:write('/cluster_ids', scores);
myFile:write('/qids', qids);
myFile:close()