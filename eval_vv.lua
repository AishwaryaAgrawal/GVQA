require 'nn'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'misc.OuterProd'
cjson=require('cjson')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate the Visual Verifier (VV) module in GVQA')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_data_folder','processed_inputs/','path to the data folder')
cmd:option('-input_question_classifier_output','predictions/question_classifier_results.h5', 'path to file containing yes no output')

cmd:option('-input_h5','processed_question_for_vv_and_ap.h5','path to the h5file containing the image feature for train')
cmd:option('-input_json','input_json_for_vv_and_ap.json','path to the json file containing additional info and vocab')

cmd:option('-input_h5_vcc_test', 'predictions/vcc_results.h5', 'path to vcc predictions')

cmd:option('-model_path', 'models/', 'path to folder containing different models')
cmd:option('-model_name', 'model_vv.t7', 'name of the model checkpoint to initialize model weights from. Empty = don\'t')

cmd:option('-out_path', 'predictions/', 'path to save output json file')

-- Model parameter settings
cmd:option('-batch_size', 1024,'batch_size for each iterations')
cmd:option('-num_visual_concepts', 1958, 'size of fc1')
cmd:option('-ques_concept_encoding_size', 300, 'encoding size for question concepts from yes no')
cmd:option('-combine_ce_vcc', 'pointwise', 'how to combine ce and vcc features. concat, pointwise, outerprod, concat_pointwise')
cmd:option('-reduce_vcc', 0, 'whether to reduce vcc (1) or not (0) before performing outerprod')
cmd:option('-reduced_vcc_size', 100, 'size of the reduced vcc for outer product with ce')
cmd:option('-num_last_fc_outerprod', 2, 'number of fc after outerprod')
cmd:option('-outerprod_fc1_size', 2000, 'size of the first hidden layer after outerprod')
cmd:option('-num_last_fc_concat', 2, 'number of fc after concatenation')
cmd:option('-concat_fc1_size', 1500, 'size of the first hidden layer after concatenation')
cmd:option('-concat_fc2_size', 1000, 'size of the second hidden layer after concatenation')
cmd:option('-num_ce_embed_fc', 2, 'number of fc for embedding ce')
cmd:option('-ce_embed_fc1_size', 512, 'size of fc1 for embedding ce')
cmd:option('-ce_embed_fc2_size', 1024, 'size of fc1 for embedding ce')
cmd:option('-num_last_fc', 2, 'number of fc after pointwise multiplication')
cmd:option('-fc1_yes_no_ans_size', 256, 'size of fc1 for answering module for non yes no')
cmd:option('-num_output_yes_no', 2, 'number of output answers for non yes no')

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

----------------------------------------------------------------------
-- Loading Dataset
----------------------------------------------------------------------
print('DataLoader loading h5 file: ', opt.input_h5)

local file = io.open(opt.input_data_folder .. opt.input_json, 'r')
local text = file:read()
file:close()
json_file_orig = cjson.decode(text)

local dataset = {}

print('DataLoader loading question classifier output')
local h5_file = hdf5.open(opt.input_question_classifier_output, 'r')
local yes_no_out_qids = h5_file:read('/qids'):all() 
local yes_no_out_pred = h5_file:read('/pred'):all() 
h5_file:close()

local h5_file = hdf5.open(opt.input_data_folder .. opt.input_h5, 'r')
local y_test_non_yes_no = h5_file:read('/answers_test_non_yes_no'):all()
local qids_test_non_yes_no = h5_file:read('/question_id_test_non_yes_no'):all()
local y_test_yes_no = h5_file:read('/answers_test_yes_no'):all()
local ques_concept_test_yes_no = h5_file:read('/ques_concept_test_yes_no'):all()
local qids_test_yes_no = h5_file:read('/question_id_test_yes_no'):all()
for i = 1, y_test_non_yes_no:size(1) do
  y_test_non_yes_no[i] = opt.num_output_yes_no + 1
end
local num_yes_no = y_test_yes_no:size(1)
local num_non_yes_no = y_test_non_yes_no:size(1)
local num_total = num_non_yes_no + num_yes_no
dataset['y_test'] = torch.IntTensor(num_total)
dataset['qids_test'] = torch.IntTensor(num_total)
dataset['x_ce_test'] = torch.Tensor(num_total, ques_concept_test_yes_no:size(2))
for i = 1, num_total do
  if i > num_yes_no then
    dataset['y_test'][i] = y_test_non_yes_no[i - num_yes_no]
    dataset['qids_test'][i] = qids_test_non_yes_no[i - num_yes_no]
  else
    dataset['y_test'][i] = y_test_yes_no[i]
    dataset['qids_test'][i] = qids_test_yes_no[i]
    dataset['x_ce_test'][i] = ques_concept_test_yes_no[i]
  end
end
h5_file:close() 

local h5_file = hdf5.open(opt.input_h5_vcc_test, 'r')
dataset['x_vcc_test'] = h5_file:read('/visual_concept_scores'):all()
local qids_vcc_test = h5_file:read('/qids'):all()
h5_file:close()

print('fetching non yes-no data')
function align_data(data, data_qids, ref, ref_qids)
  local qids_to_idx = {}
  for i = 1, data_qids:size(1) do
      qids_to_idx[data_qids[i]] = i
  end

  local des_inds = {}
  local num_data_points = ref_qids:size(1)
  local count = 0
  for i = 1, num_data_points do
      if ref[i] == 1 then
          count = count + 1
          des_inds[count] = qids_to_idx[ref_qids[i]]
      end
  end

  des_inds = torch.LongTensor(des_inds)
  aligned_data = data:index(1, des_inds)

  return aligned_data
end

dataset['x_ce_test'] = align_data(dataset['x_ce_test'], dataset['qids_test'], yes_no_out_pred, yes_no_out_qids)
dataset['x_vcc_test'] = align_data(dataset['x_vcc_test'], qids_vcc_test, yes_no_out_pred, yes_no_out_qids)
dataset['y_test'] = align_data(dataset['y_test'], dataset['qids_test'], yes_no_out_pred, yes_no_out_qids)
dataset['qids_test'] = align_data(dataset['qids_test'], dataset['qids_test'], yes_no_out_pred, yes_no_out_qids)

local num_test = dataset['qids_test']:size(1)

print(dataset['x_ce_test']:size())
print(dataset['x_vcc_test']:size())
print(dataset['y_test']:size())
print(dataset['qids_test']:size())

collectgarbage()

------------------------------------------------------------------------
-- Design Parameters and Network Definitions
------------------------------------------------------------------------

print('Building the model...')

-- answering module for non yes/no

if opt.combine_ce_vcc == 'pointwise' then
    if opt.num_ce_embed_fc == 1 then
        non_yes_no_embed_ce = nn.Sequential()
                               :add(nn.Linear(opt.ques_concept_encoding_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    elseif opt.num_ce_embed_fc == 2 then
        non_yes_no_embed_ce = nn.Sequential()
                               :add(nn.Linear(opt.ques_concept_encoding_size, opt.ce_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    elseif opt.num_ce_embed_fc == 3 then
        non_yes_no_embed_ce = nn.Sequential()
                               :add(nn.Linear(opt.ques_concept_encoding_size, opt.ce_embed_fc1_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc1_size, opt.ce_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    end

    non_yes_no_embed_vcc = nn.Sequential()
                           :add(nn.Identity())

    non_yes_no_embed_combined = nn.ParallelTable()
                                :add(non_yes_no_embed_ce)
                                :add(non_yes_no_embed_vcc)   
    if opt.num_last_fc == 1 then
        model = nn.Sequential()
                :add(non_yes_no_embed_combined)
                :add(nn.CMulTable())
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts, opt.num_output_yes_no))
    elseif opt.num_last_fc == 2 then
        model = nn.Sequential()
                :add(non_yes_no_embed_combined)
                :add(nn.CMulTable())
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts, opt.fc1_yes_no_ans_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.fc1_yes_no_ans_size, opt.num_output_yes_no))
    end

elseif opt.combine_ce_vcc == 'concat' then
    if opt.num_last_fc == 2 then
        model = nn.Sequential()
                :add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.ques_concept_encoding_size, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_yes_no))
    elseif opt.num_last_fc == 3 then
        model = nn.Sequential()
                :add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.ques_concept_encoding_size, opt.concat_fc1_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc1_size, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_yes_no))
    end

elseif opt.combine_ce_vcc == 'concat_pointwise' then

    if opt.num_ce_embed_fc == 2 then
        non_yes_no_embed_ce = nn.Sequential()
                               :add(nn.Linear(opt.ques_concept_encoding_size, opt.ce_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    elseif opt.num_ce_embed_fc == 3 then
        non_yes_no_embed_ce = nn.Sequential()
                               :add(nn.Linear(opt.ques_concept_encoding_size, opt.ce_embed_fc1_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc1_size, opt.ce_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.ce_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    end

    non_yes_no_embed_vcc = nn.Sequential()
                           :add(nn.Identity())

    non_yes_no_embed_combined = nn.ParallelTable()
                                :add(non_yes_no_embed_ce)
                                :add(non_yes_no_embed_vcc)   

    pointwise_net = nn.Sequential()
                    :add(non_yes_no_embed_combined)
                    :add(nn.CMulTable())
                    :add(nn.Dropout(0.5))

    concat_net = nn.ConcatTable()
                 :add(nn.SelectTable(1))
                 :add(pointwise_net)
                 :add(nn.SelectTable(2))
    
    model = nn.Sequential()
            :add(concat_net) 
          
    if opt.num_last_fc == 2 then
                model:add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.ques_concept_encoding_size+opt.num_visual_concepts, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_yes_no))
    elseif opt.num_last_fc == 3 then
                model:add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.ques_concept_encoding_size+opt.num_visual_concepts, opt.concat_fc1_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc1_size, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_yes_no))
    end

elseif opt.combine_ce_vcc == 'outerprod' then
    embed_ce = nn.Sequential()
                :add(nn.Identity())

    if opt.reduce_vcc == 1 then            
        embed_vcc = nn.Sequential()
                    :add(nn.Linear(opt.num_visual_concepts, opt.reduced_vcc_size))
                    :add(nn.Tanh())
                    :add(nn.Dropout(0.5))
    else
        embed_vcc = nn.Sequential()
                    :add(nn.Identity())
    end

    embed_combined = nn.ParallelTable()
                     :add(embed_ce)
                     :add(embed_vcc)  

    model = nn.Sequential()
            :add(embed_combined)
            :add(nn.OuterProd())
            :add(nn.View(-1):setNumInputDims(2)) -- convert the outerproduct matrix to vector
            :add(nn.Dropout(0.5))

    if opt.num_last_fc_outerprod == 1 then
        if opt.reduce_vcc == 1 then
            model:add(nn.Linear(opt.ques_concept_encoding_size*opt.reduced_vcc_size, opt.num_output_yes_no))
        else
            model:add(nn.Linear(opt.ques_concept_encoding_size*opt.num_visual_concepts, opt.num_output_yes_no))
        end

    elseif opt.num_last_fc_outerprod == 2 then
        if opt.reduce_vcc == 1 then
            model:add(nn.Linear(opt.ques_concept_encoding_size*opt.reduced_vcc_size, opt.outerprod_fc1_size))
        else
            model:add(nn.Linear(opt.ques_concept_encoding_size*opt.num_visual_concepts, opt.outerprod_fc1_size))
        end
        model:add(nn.Tanh())
             :add(nn.Dropout(0.5))
             :add(nn.Linear(opt.outerprod_fc1_size, opt.num_output_yes_no))
    end
end
 
if opt.gpuid >= 0 then
    print('shipped data function to cuda...')
    model = model:cuda()
end

-- setting to evaluation
model:evaluate()

model_w, model_dw = model:getParameters()
print(model_w:size())

-- loading the model
model_param = torch.load(opt.model_path .. opt.model_name)
print(model_param['model_w']:size())
model_w:copy(model_param['model_w'])

function dataset:next_batch_test(s, e)

    local batch_size = e - s + 1
    local qinds = torch.LongTensor(batch_size):fill(0)

    for i = 1, batch_size do
        qinds[i] = s+i-1
    end

    fv_ce_test = dataset['x_ce_test']:index(1,qinds)
    fv_vcc_test = dataset['x_vcc_test']:index(1,qinds)

    -- ship to gpu
    if opt.gpuid >= 0 then
        fv_ce_test = fv_ce_test:cuda()
        fv_vcc_test = fv_vcc_test:cuda()
    end

    return fv_ce_test, fv_vcc_test
end

function forward(s, e)

    -- get a batch
    local fv_ce, fv_vcc = dataset:next_batch_test(s, e)

    -- model forward pass
    local scores = model:forward({fv_ce, fv_vcc})

    return scores:double()

end

function writeAll(file,data)
    local f = io.open(file, "w")
    f:write(data)
    f:close()
end

function saveJson(fname,t)
    return writeAll(fname,cjson.encode(t))
end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------

num_testing_points = dataset['x_ce_test']:size(1)
gt_ind = dataset['y_test']
qids = dataset['qids_test']

pred_scores = torch.Tensor(num_testing_points, opt.num_output_yes_no)

for i = 1, num_testing_points, opt.batch_size do
    xlua.progress(i, num_testing_points)
    r = math.min(i + opt.batch_size - 1, num_testing_points)
    pred_scores[{{i, r},{}}] = forward(i, r)
end

_, pred_ind = pred_scores:max(2)

acc = 100*torch.sum(torch.eq(pred_ind:int(), gt_ind))/num_testing_points

response = {}
for i = 1, num_testing_points do
    local old_ans_label = json_file_orig['new_to_old_ans_label_yes_no'][tostring(pred_ind[{i, 1}])]
    table.insert(response, {question_id = qids[i], answer = json_file_orig['ix_to_ans'][tostring(old_ans_label)]})
end

paths.mkdir(opt.out_path)
saveJson(opt.out_path .. 'vv_results.json', response)

print("1/0 accuracy is " .. acc)