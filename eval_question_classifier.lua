require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'rnn'
cjson = require('cjson')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate the question classifier module in GVQA')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_data_folder','/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/processed_inputs/','path to the data folder')
cmd:option('-input_ques_h5','processed_question_for_question_classifier.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','input_json_for_question_classifier.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/models/', 'path to folder containing different models')
cmd:option('-model_name', 'model_question_classifier.t7', 'name of the model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/predictions/', 'path to save predictions files')

-- Model parameter settings
cmd:option('-question_word_encoding','glove','glove or one-hot')
cmd:option('-batch_size', 128,'batch_size for each iterations')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size', 512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',1,'number of the rnn layer')
cmd:option('-num_output', 1, 'number of output labels')

-- Miscellaneous
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-num_ques', -1, 'number of questions to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------

local file = io.open(opt.input_data_folder .. opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_data_folder .. opt.input_ques_h5, 'r')

if opt.question_word_encoding == 'one-hot' then
    local count = 0
    for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
    vocabulary_size_q=count
elseif opt.question_word_encoding == 'glove' then
    ques_word_embedding_dim = 300
end

print("preparing test data")
local dataset_test = {}

dataset_test['question_yes_no'] = h5_file:read('/ques_test_yes_no'):all()
dataset_test['lengths_q_yes_no'] = h5_file:read('/ques_length_test_yes_no'):all()
dataset_test['question_id_yes_no'] = h5_file:read('/question_id_test_yes_no'):all()

dataset_test['question_non_yes_no'] = h5_file:read('/ques_test_non_yes_no'):all()
dataset_test['lengths_q_non_yes_no'] = h5_file:read('/ques_length_test_non_yes_no'):all()
dataset_test['question_id_non_yes_no'] = h5_file:read('/question_id_test_non_yes_no'):all()

local num_yes_no = dataset_test['lengths_q_yes_no']:size(1)
local num_non_yes_no = dataset_test['lengths_q_non_yes_no']:size(1)
local num_total = num_yes_no + num_non_yes_no

dataset_test['labels'] = torch.IntTensor(num_total)
dataset_test['question'] = torch.Tensor(num_total, dataset_test['question_yes_no']:size(2), dataset_test['question_yes_no']:size(3))
dataset_test['lengths_q'] = torch.IntTensor(num_total)
dataset_test['qids'] = torch.IntTensor(num_total)

for i = 1, num_total do

    if i > num_yes_no then
        dataset_test['question'][i] = dataset_test['question_non_yes_no'][i - num_yes_no]
        dataset_test['lengths_q'][i] = dataset_test['lengths_q_non_yes_no'][i - num_yes_no]
        dataset_test['qids'][i] = dataset_test['question_id_non_yes_no'][i - num_yes_no]
        dataset_test['labels'][i] = 0
       
    else 
        dataset_test['question'][i] = dataset_test['question_yes_no'][i]
        dataset_test['lengths_q'][i] = dataset_test['lengths_q_yes_no'][i]
        dataset_test['qids'][i] = dataset_test['question_id_yes_no'][i]
        dataset_test['labels'][i] = 1
     
    end
end

dataset_test['question'] = right_align(dataset_test['question'],dataset_test['lengths_q'])
h5_file:close()
collectgarbage()

------------------------------------------------------------------------
-- Design Parameters and Network Definitions
------------------------------------------------------------------------

print('Building the model...')

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
     :add(nn.Linear(opt.rnn_size, opt.num_output))

sigmoid_net = nn.Sigmoid()

if opt.gpuid >= 0 then
    print('shipped data function to cuda...')
    rnn_q = rnn_q:cuda()
    sigmoid_net = sigmoid_net:cuda()
end

-- setting to evaluation
rnn_q:evaluate()
sigmoid_net:evaluate()

-- loading the model
rnn_q_w, rnn_q_dw = rnn_q:getParameters()
model_param = torch.load(opt.model_path .. opt.model_name)
rnn_q_w:copy(model_param['rnn_q_w'])

sizes = {rnn_q_w:size(1)}

function dataset:next_batch_test(s, e)

    local batch_size = e - s + 1
    local qinds = torch.LongTensor(batch_size):fill(0)

    for i = 1, batch_size do
        qinds[i] = s+i-1
    end

    fv_q_test = dataset_test['question']:index(1,qinds)
    qids_test = dataset_test['qids']:index(1,qinds)

    -- ship to gpu
    if opt.gpuid >= 0 then
        fv_q_test = fv_q_test:cuda()
    end

    return fv_q_test, qids_test
end

function forward(s, e)

    -- get a batch
    local fv_q, qids = dataset:next_batch_test(s, e)

    -- model forward pass
    local scores = rnn_q:forward(fv_q)
    
    -- sigmoid forward pass
    local sigmoid_scores = sigmoid_net:forward(scores)
    
    return sigmoid_scores:double(), qids

end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------

num_testing_points = dataset_test['question']:size(1)

pred_scores = torch.Tensor(num_testing_points, opt.num_output)
qids = torch.LongTensor(num_testing_points)

for i = 1, num_testing_points, opt.batch_size do
    xlua.progress(i, num_testing_points)
    r = math.min(i + opt.batch_size - 1, num_testing_points)
    pred_scores[{{i, r},{}}], qids[{{i, r}}] = forward(i, r)
end

gt_ind = dataset_test['labels']

acc = 100*torch.sum(torch.eq(pred_scores:gt(0.5):int(), gt_ind))/num_testing_points
print('saving predictions')
local myFile = hdf5.open(opt.out_path .. 'question_classifier_results.h5', 'w');
myFile:write('/pred', pred_scores:gt(0.5):int():view(-1));
myFile:write('/qids', qids);
myFile:close()
print("accuracy is " .. acc)
