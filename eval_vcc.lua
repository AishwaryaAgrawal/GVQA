require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'rnn'
cjson = require('cjson')
require 'loadcaffe'
require 'image'
san = require 'san.lua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate the VCC module in GVQA')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_data_folder','processed_inputs/','path to the data folder')
cmd:option('-input_img_h5','processed_test_image_for_vcc.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','processed_question_for_vcc.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','input_json_for_vcc.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'models/', 'path to folder containing different models')
cmd:option('-model_name', 'model_vcc.t7', 'name of the model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'predictions/', 'path to save output json file')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-question_word_encoding','glove','glove or one-hot')
cmd:option('-batch_size',1024,'batch_size for each iterations')
cmd:option('-input_encoding_size', 200, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',1,'number of the rnn layer')
cmd:option('-common_embedding_size', 512, 'size of the common embedding vector')
cmd:option('-num_last_fc', 2, 'num of fc in vcc')
cmd:option('-fc1_size', 1000, 'size of fc1')
cmd:option('-fc2_size', 1000, 'size of fc1')
cmd:option('-non_linearity','tanh','tanh or relu for fcs in vcc')
cmd:option('-num_visual_concepts', 1958, 'number of visual concepts for VCC classes')
cmd:option('-img_tr_size', 512, 'size of image feature projection')  -- this should be same as rnn_size
cmd:option('-num_attention_layer', 2, 'number of attention layers')

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

print('DataLoader loading json file: ', opt.input_json)
local file = io.open(opt.input_data_folder .. opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_data_folder .. opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_test'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
dataset['img_list'] = h5_file:read('/img_pos_test'):all()
dataset['ques_id'] = h5_file:read('/question_id_test'):all()

h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_data_folder .. opt.input_img_h5, 'r')
fv_im_train = h5_file:read('/images_test'):all()

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
        :add(nn.SplitTable(1, 2))
        :add(nn.Sequencer(nn.FastLSTM(opt.input_encoding_size, opt.rnn_size):maskZero(1)))
end

if opt.rnn_layer > 1 then
    for i = 1, opt.rnn_layer - 1 do
        rnn_q:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size):maskZero(1)))
    end
end

rnn_q:add(nn.SelectTable(-1))

san_net = san.n_attention_layer(512, opt.img_tr_size, opt.rnn_size, opt.common_embedding_size, opt.num_attention_layer, opt.num_last_fc, opt.fc1_size, opt.fc2_size, opt.non_linearity, opt.num_visual_concepts)

sigmoid_net = nn.Sigmoid()

if opt.gpuid >= 0 then
    print('shipped data function to cuda...')
    rnn_q = rnn_q:cuda()
    san_net = san_net:cuda()
    sigmoid_net = sigmoid_net:cuda()
end

-- setting to evaluation
rnn_q:evaluate()
san_net:evaluate()
sigmoid_net:evaluate()

rnn_q_w, rnn_q_dw = rnn_q:getParameters()
san_net_w, san_net_dw = san_net:getParameters()

-- loading the model
model_param = torch.load(opt.model_path .. opt.model_name)
rnn_q_w:copy(model_param['rnn_q_w'])
san_net_w:copy(model_param['san_net_w'])

sizes = {rnn_q_w:size(1), san_net_w:size(1)}

------------------------------------------------------------------------
-- Grab Next Batch --
------------------------------------------------------------------------

function dataset:next_batch_test(s, e)

    local batch_size = e - s + 1
    local qinds = torch.LongTensor(batch_size):fill(0)
    local iminds = torch.LongTensor(batch_size):fill(0)

    for i = 1, batch_size do
        qinds[i] = s+i-1
        iminds[i] = dataset['img_list'][qinds[i]]
    end

    local fv_q = dataset['question']:index(1,qinds)

    local fv_im = fv_im_train:index(1, iminds)
    local qids = dataset['ques_id']:index(1,qinds)

    -- ship to gpu
    if opt.gpuid >= 0 then
        fv_q = fv_q:cuda()
        fv_im = fv_im:cuda()
    end

    return fv_q, fv_im, qids
end

function forward(s,e)
    -- get a batch
    local fv_q, fv_im, qids = dataset:next_batch_test(s, e)

    -- LookupTable + RNN forward pass
    local tv_q = rnn_q:forward(fv_q)

    -- -- SAN forward pass
    local scores = san_net:forward({fv_im, tv_q})

     -- sigmoid forward pass
    local sigmoid_scores = sigmoid_net:forward(scores)

    -- For SAN-2, layers 22, 35 are softmax
    -- For SAN-1, layer 19 is softmax
    local maps, maps_1, maps_2
    if opt.num_attention_layer == 1 then
        maps = san_net.modules[19].output
        maps = maps:float()
        maps = nn.View(-1, 14, 14):forward(maps)
        return sigmoid_scores:double(), maps, qids
    elseif opt.num_attention_layer == 2 then
        maps_1 = san_net.modules[22].output
        maps_2 = san_net.modules[35].output
        maps_1 = maps_1:float()
        maps_1 = nn.View(-1, 14, 14):forward(maps_1)
        maps_2 = maps_2:float()
        maps_2 = nn.View(-1, 14, 14):forward(maps_2)
        return sigmoid_scores:double(), maps_1, maps_2, qids
    else
        return sigmoid_scores:double(), qids
    end
end


-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------

nqs = dataset['question']:size(1)
if opt.num_ques >= 0 then
    nqs = opt.num_ques
end
sigmoid_scores = torch.Tensor(nqs, opt.num_visual_concepts)
maps_1 = torch.Tensor(nqs, 14, 14)
maps_2 = torch.Tensor(nqs, 14, 14)
qids = torch.LongTensor(nqs)

for i = 1, nqs, opt.batch_size do
    xlua.progress(i, nqs)
    r = math.min(i + opt.batch_size - 1, nqs)
    if opt.num_attention_layer == 1 then
        sigmoid_scores[{{i, r},{}}], maps_1[{{i, r},{}}], qids[{{i, r}}] = forward(i, r)
    elseif opt.num_attention_layer == 2 then
        sigmoid_scores[{{i, r},{}}], maps_1[{{i, r},{}}], maps_2[{{i, r},{}}], qids[{{i, r}}] = forward(i, r)
    else
        sigmoid_scores[{{i, r},{}}], qids[{{i, r}}] = forward(i, r)
    end
end

print('saving predicted visual concept scores')
paths.mkdir(opt.out_path)
local myFile = hdf5.open(opt.out_path .. 'vcc_results.h5', 'w');
myFile:write('/visual_concept_scores', sigmoid_scores);
myFile:write('/qids', qids);
myFile:close()

-- Save maps as hdf5
h5_file = hdf5.open(opt.out_path .. 'vcc_attention_maps.h5', 'w')
if opt.num_attention_layer == 1 then
    h5_file:write('/maps_att1', maps_1)
elseif opt.num_attention_layer == 2 then
    h5_file:write('/maps_att1', maps_1)
    h5_file:write('/maps_att2', maps_2)
end
h5_file:close()
