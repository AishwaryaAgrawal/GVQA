require 'nn';
require 'nngraph';

local san = {}

function san.n_attention_layer(img_feat_size, img_tr_size, rnn_size, common_embedding_size, num_attention_layer, num_last_fc, fc1_size, fc2_size, non_linearity, num_visual_concepts)
    local inputs, outputs = {}, {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local img_feat = inputs[1]
    local ques_feat = inputs[2]

    local u = ques_feat
    local img_tr = nn.Dropout(0.5)(nn.Tanh()(nn.View(-1, 196, img_tr_size)(nn.Linear(img_feat_size, img_tr_size)(nn.View(img_feat_size):setNumInputDims(2)(img_feat)))))

    for i = 1, num_attention_layer do

        -- linear layer: 14x14x512 -> 14x14x512
        local img_common = nn.View(-1, 196, common_embedding_size)(nn.Linear(img_tr_size, common_embedding_size)(nn.View(-1, img_tr_size)(img_tr)))

        -- replicate lstm state 196 times
        local ques_common = nn.Linear(rnn_size, common_embedding_size)(u)
        local ques_repl = nn.Replicate(196, 2)(ques_common)

        -- add image and question features (both 196x512)
        local img_ques_common = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({img_common, ques_repl})))
        local h = nn.Linear(common_embedding_size, 1)(nn.View(-1, common_embedding_size)(img_ques_common))
        local p = nn.SoftMax()(nn.View(-1, 196)(h))

        -- weighted sum of image features
        local p_att = nn.View(1, -1):setNumInputDims(1)(p)
        local img_tr_att = nn.MM(false, false)({p_att, img_tr})
        local img_tr_att_feat = nn.View(-1, img_tr_size)(img_tr_att)

        -- add image feature vector and question vector
        if i == num_attention_layer then
            u = img_tr_att_feat
        else
            u = nn.CAddTable()({img_tr_att_feat, u})
        end

    end

    if num_last_fc == 2 then
        if non_linearity == 'tanh' then
            local o = nn.Linear(fc1_size,num_visual_concepts)(nn.Dropout(0.5)(nn.Tanh()(nn.Linear(img_tr_size, fc1_size)(nn.Dropout(0.5)(u)))))
            table.insert(outputs, o)
        elseif non_linearity == 'relu' then
            local o = nn.Linear(fc1_size,num_visual_concepts)(nn.Dropout(0.5)(nn.ReLU()(nn.Linear(img_tr_size, fc1_size)(nn.Dropout(0.5)(u)))))
            table.insert(outputs, o)
        end
    elseif num_last_fc == 3 then
        if non_linearity == 'tanh' then
            local o = nn.Linear(fc2_size,num_visual_concepts)(nn.Dropout(0.5)(nn.Tanh()(nn.Linear(fc1_size,fc2_size)(nn.Dropout(0.5)(nn.Tanh()(nn.Linear(img_tr_size, fc1_size)(nn.Dropout(0.5)(u))))))))
            table.insert(outputs, o)
        elseif non_linearity == 'relu' then
            local o = nn.Linear(fc2_size,num_visual_concepts)(nn.Dropout(0.5)(nn.ReLU()(nn.Linear(fc1_size,fc2_size)(nn.Dropout(0.5)(nn.ReLU()(nn.Linear(img_tr_size, fc1_size)(nn.Dropout(0.5)(u))))))))
            table.insert(outputs, o)
        end
    end

    return nn.gModule(inputs, outputs)
end

return san
