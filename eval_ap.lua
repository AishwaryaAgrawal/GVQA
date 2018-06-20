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
cmd:text('Evaluate the Answer Predictor (AP) module in GVQA')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_data_folder','/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/processed_inputs/','path to the data folder')
cmd:option('-input_question_classifier_output','/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/predictions/question_classifier_results.h5', 'path to file containing yes no output')

cmd:option('-input_h5','processed_question_for_vv_and_ap.h5','path to the h5file containing the image feature for train')
cmd:option('-input_json','input_json_for_vcc.json','path to json file containing information about alignment between acp and vcc clusters')

cmd:option('-input_h5_acp_test', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/predictions/acp_results.h5', 'path to the h5file containing the acp predictions for test')
cmd:option('-input_h5_vcc_test', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/predictions/vcc_results.h5', 'path to the h5file containing the vcc predictions for test')

cmd:option('-input_json_orig','input_json_for_vv_and_ap.json','path to the json file containing additional info and vocab')

cmd:option('-model_path', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/models/', 'path to folder containing different models')
cmd:option('-model_name', 'model_ap.t7', 'name of the model checkpoint to initialize model weights from. Empty = don\'t')

cmd:option('-out_path', '/srv/share/aagrawal307/compositionality/vqa/vqa_compositionality/to_release/predictions/', 'path to save output json file')
cmd:option('-num_visual_concepts', 1958, 'size of fc1') -- change this depending on which VCC is being used

-- Model parameter settings
cmd:option('-batch_size', 1024,'batch_size for each iterations')
cmd:option('-num_answer_clusters', 50, 'size of fc1')
cmd:option('-combine_acp_vcc', 'pointwise', 'how to combine acp and vcc features. concat, pointwise, outerprod, concat_pointwise')
cmd:option('-reduce_vcc', 0, 'whether to reduce vcc (1) or not (0) before performing outerprod')
cmd:option('-reduced_vcc_size', 100, 'size of the reduced vcc for outer product with acp')
cmd:option('-num_last_fc_outerprod', 2, 'number of fc after outerprod')
cmd:option('-outerprod_fc1_size', 2000, 'size of the first hidden layer after outerprod')
cmd:option('-num_last_fc_concat', 2, 'number of fc after concatenation')
cmd:option('-concat_fc1_size', 1500, 'size of the first hidden layer after concatenation')
cmd:option('-concat_fc2_size', 1000, 'size of the second hidden layer after concatenation')
cmd:option('-num_acp_embed_fc', 2, 'number of fc for embedding acp')
cmd:option('-acp_embed_fc1_size', 512, 'size of fc1 for embedding acp')
cmd:option('-acp_embed_fc2_size', 1024, 'size of fc1 for embedding acp')
cmd:option('-num_last_fc', 3, 'number of fc after pointwise multiplication')
cmd:option('-fc1_non_yes_no_ans_size', 1000, 'size of fc1 for answering module for non yes no')
cmd:option('-fc2_non_yes_no_ans_size', 1000, 'size of fc1 for answering module for non yes no')
cmd:option('-num_output_non_yes_no', 998, 'number of output answers for non yes no')

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
local map = {}
local file = io.open(opt.input_data_folder .. opt.input_json_orig, 'r')
local text = file:read()
file:close()
json_file_orig = cjson.decode(text)
map['new_to_old_ans_label_non_yes_no'] = json_file_orig['new_to_old_ans_label_non_yes_no']
map['ix_to_ans'] = json_file_orig['ix_to_ans']

json_file_orig = nil

local file = io.open(opt.input_data_folder .. opt.input_json, 'r')
local text = file:read()
file:close()
input_json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_h5)
local dataset = {}

print('DataLoader loading yes no classifier output')
local h5_file = hdf5.open(opt.input_question_classifier_output, 'r')
local yes_no_out_qids = h5_file:read('/qids'):all() 
local yes_no_out_pred = h5_file:read('/pred'):all() 
h5_file:close()

local h5_file = hdf5.open(opt.input_data_folder .. opt.input_h5, 'r')
local y_test_non_yes_no = h5_file:read('/answers_test_non_yes_no'):all()
local qids_test_non_yes_no = h5_file:read('/question_id_test_non_yes_no'):all()
local y_test_yes_no = h5_file:read('/answers_test_yes_no'):all()
local qids_test_yes_no = h5_file:read('/question_id_test_yes_no'):all()
for i = 1, y_test_yes_no:size(1) do
  y_test_yes_no[i] = opt.num_output_non_yes_no + 1
end
local num_yes_no = y_test_yes_no:size(1)
local num_non_yes_no = y_test_non_yes_no:size(1)
local num_total = num_non_yes_no + num_yes_no
dataset['y_test'] = torch.IntTensor(num_total)
dataset['qids_test'] = torch.IntTensor(num_total)
for i = 1, num_total do
  if i > num_yes_no then
    dataset['y_test'][i] = y_test_non_yes_no[i - num_yes_no]
    dataset['qids_test'][i] = qids_test_non_yes_no[i - num_yes_no]
  else
    dataset['y_test'][i] = y_test_yes_no[i]
    dataset['qids_test'][i] = qids_test_yes_no[i]
  end
end
h5_file:close()

local h5_file = hdf5.open(opt.input_h5_acp_test, 'r')
dataset['x_acp_test'] = h5_file:read('/cluster_ids'):all()
local qids_acp_test = h5_file:read('/qids'):all()
h5_file:close()

local h5_file = hdf5.open(opt.input_h5_vcc_test, 'r')
dataset['x_vcc_test'] = h5_file:read('/visual_concept_scores'):all()
local qids_vcc_test = h5_file:read('/qids'):all()
h5_file:close()

---------

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
      if ref[i] == 0 then
          count = count + 1
          des_inds[count] = qids_to_idx[ref_qids[i]]
      end
  end

  des_inds = torch.LongTensor(des_inds)
  aligned_data = data:index(1, des_inds)

  return aligned_data
end

dataset['x_acp_test'] = align_data(dataset['x_acp_test'], qids_acp_test, yes_no_out_pred, yes_no_out_qids)
dataset['x_vcc_test'] = align_data(dataset['x_vcc_test'], qids_vcc_test, yes_no_out_pred, yes_no_out_qids)
dataset['y_test'] = align_data(dataset['y_test'], dataset['qids_test'], yes_no_out_pred, yes_no_out_qids)
dataset['qids_test'] = align_data(dataset['qids_test'], dataset['qids_test'], yes_no_out_pred, yes_no_out_qids)
------

print('passing x_acp through softmax')
dataset['x_acp_test'] = nn.SoftMax():forward(dataset['x_acp_test'])

local num_test = dataset['qids_test']:size(1)

function align_acp(acp_features, visual_concept_to_cluster_label, num_visual_concepts)
  aligned_acp_features = torch.Tensor(acp_features:size(1), num_visual_concepts)
  for i = 1, acp_features:size(1) do
    for j = 1, num_visual_concepts do
      aligned_acp_features[i][j] = acp_features[i][visual_concept_to_cluster_label[j]]
    end
  end
  return aligned_acp_features
end

print("aligning acp feature vector as per vcc")

local num_visual_concepts = opt.num_visual_concepts

print("creating mapping")
local visual_concept_to_cluster_label = {} 
for j = 1, num_visual_concepts do
  local visual_concept = input_json_file['ix_to_visual_concepts'][tostring(j)]
  local cluster_label = input_json_file['visual_concept_to_cluster_labels'][visual_concept] + 1 -- because in the json file, clusters are 0-indexed
  visual_concept_to_cluster_label[j] = cluster_label
end

print("aligning")
dataset['x_acp_test'] = align_acp(dataset['x_acp_test'], visual_concept_to_cluster_label, num_visual_concepts)

collectgarbage()

------------------------------------------------------------------------
-- Design Parameters and Network Definitions
------------------------------------------------------------------------

print('Building the model...')

-- answering module for non yes/no

if opt.combine_acp_vcc == 'pointwise' then   
    if opt.num_last_fc == 1 then
        model = nn.Sequential()
                :add(nn.CMulTable())
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts, opt.num_output_non_yes_no))
    elseif opt.num_last_fc == 2 then
        model = nn.Sequential()
                :add(nn.CMulTable())
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts, opt.fc1_non_yes_no_ans_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.fc1_non_yes_no_ans_size, opt.num_output_non_yes_no))
    elseif opt.num_last_fc == 3 then
        model = nn.Sequential()
            :add(nn.CMulTable())
            :add(nn.Tanh())
            :add(nn.Dropout(0.5))
            :add(nn.Linear(opt.num_visual_concepts, opt.fc1_non_yes_no_ans_size))
            :add(nn.Tanh())
            :add(nn.Dropout(0.5))
            :add(nn.Linear(opt.fc1_non_yes_no_ans_size, opt.fc2_non_yes_no_ans_size))
            :add(nn.Tanh())
            :add(nn.Dropout(0.5))
            :add(nn.Linear(opt.fc2_non_yes_no_ans_size, opt.num_output_non_yes_no))
    end

elseif opt.combine_acp_vcc == 'concat' then
    if opt.num_last_fc == 2 then
        model = nn.Sequential()
                :add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.num_answer_clusters, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_non_yes_no))
    elseif opt.num_last_fc == 3 then
        model = nn.Sequential()
                :add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.num_answer_clusters, opt.concat_fc1_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc1_size, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_non_yes_no))
    end

elseif opt.combine_acp_vcc == 'concat_pointwise' then

    if opt.num_acp_embed_fc == 2 then
        non_yes_no_embed_acp = nn.Sequential()
                               :add(nn.Linear(opt.num_answer_clusters, opt.acp_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.acp_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    elseif opt.num_acp_embed_fc == 3 then
        non_yes_no_embed_acp = nn.Sequential()
                               :add(nn.Linear(opt.num_answer_clusters, opt.acp_embed_fc1_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.acp_embed_fc1_size, opt.acp_embed_fc2_size))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
                               :add(nn.Linear(opt.acp_embed_fc2_size, opt.num_visual_concepts))
                               :add(nn.Tanh())
                               :add(nn.Dropout(0.5))
    end

    non_yes_no_embed_vcc = nn.Sequential()
                           :add(nn.Identity())

    non_yes_no_embed_combined = nn.ParallelTable()
                                :add(non_yes_no_embed_acp)
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
                :add(nn.Linear(opt.num_visual_concepts+opt.num_answer_clusters+opt.num_visual_concepts, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_non_yes_no))
    elseif opt.num_last_fc == 3 then
                model:add(nn.JoinTable(1,1))
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.num_visual_concepts+opt.num_answer_clusters+opt.num_visual_concepts, opt.concat_fc1_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc1_size, opt.concat_fc2_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.concat_fc2_size, opt.num_output_non_yes_no))
    end

elseif opt.combine_acp_vcc == 'outerprod' then
    embed_acp = nn.Sequential()
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
                     :add(embed_acp)
                     :add(embed_vcc)  

    model = nn.Sequential()
            :add(embed_combined)
            :add(nn.OuterProd())
            :add(nn.View(-1):setNumInputDims(2)) -- convert the outerproduct matrix to vector
            :add(nn.Dropout(0.5))

    if opt.num_last_fc_outerprod == 1 then
        if opt.reduce_vcc == 1 then
            model:add(nn.Linear(opt.num_answer_clusters*opt.reduced_vcc_size, opt.num_output_non_yes_no))
        else
            model:add(nn.Linear(opt.num_answer_clusters*opt.num_visual_concepts, opt.num_output_non_yes_no))
        end

    elseif opt.num_last_fc_outerprod == 2 then
        if opt.reduce_vcc == 1 then
            model:add(nn.Linear(opt.num_answer_clusters*opt.reduced_vcc_size, opt.outerprod_fc1_size))
        else
            model:add(nn.Linear(opt.num_answer_clusters*opt.num_visual_concepts, opt.outerprod_fc1_size))
        end
        model:add(nn.Tanh())
             :add(nn.Dropout(0.5))
             :add(nn.Linear(opt.outerprod_fc1_size, opt.num_output_non_yes_no))
    end
end
 
if opt.gpuid >= 0 then
    print('shipped data function to cuda...')
    model = model:cuda()
end

-- setting to evaluation
model:evaluate()

model_w, model_dw = model:getParameters()

-- loading the model
model_param = torch.load(opt.model_path .. opt.model_name)
model_w:copy(model_param['model_w'])

function dataset:next_batch_test(s, e)

    local batch_size = e - s + 1
    local qinds = torch.LongTensor(batch_size):fill(0)

    for i = 1, batch_size do
        qinds[i] = s+i-1
    end

    fv_acp_test = dataset['x_acp_test']:index(1,qinds)
    fv_vcc_test = dataset['x_vcc_test']:index(1,qinds)

    -- ship to gpu
    if opt.gpuid >= 0 then
        fv_acp_test = fv_acp_test:cuda()
        fv_vcc_test = fv_vcc_test:cuda()
    end

    return fv_acp_test, fv_vcc_test
end

function forward(s, e)

    -- get a batch
    local fv_acp, fv_vcc = dataset:next_batch_test(s, e)

    -- model forward pass
    local scores = model:forward({fv_acp, fv_vcc})

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

num_testing_points = dataset['x_acp_test']:size(1)
gt_ind = dataset['y_test']
qids = dataset['qids_test']

pred_scores = torch.Tensor(num_testing_points, opt.num_output_non_yes_no)

for i = 1, num_testing_points, opt.batch_size do
    xlua.progress(i, num_testing_points)
    r = math.min(i + opt.batch_size - 1, num_testing_points)
    pred_scores[{{i, r},{}}] = forward(i, r)
end

_, pred_ind = pred_scores:max(2)

paths.mkdir(opt.out_path)

myFile = hdf5.open(opt.out_path .. 'ap_results.h5', 'w')
myFile:write('/pred_scores', pred_scores)
myFile:close()

acc = 100*torch.sum(torch.eq(pred_ind:int(), gt_ind))/num_testing_points

response = {}
print(qids:size())
for i = 1, num_testing_points do
    local old_ans_label = map['new_to_old_ans_label_non_yes_no'][tostring(pred_ind[{i, 1}])]
    table.insert(response, {question_id = qids[i], answer = map['ix_to_ans'][tostring(old_ans_label)]})
end

saveJson(opt.out_path .. 'ap_results.json', response)

print("1/0 accuracy is " .. acc)