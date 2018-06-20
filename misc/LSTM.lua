--[[
Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
]]--

require 'nn'
require 'nngraph'

LSTM={};


function LSTM.lstm_conventional(input_size,rnn_size,noutput,n,dropout)
    dropout = dropout or 0 
--my wrapper
    local h_old=nn.Identity()();
	local input=nn.Identity()();
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, input) -- x
  for L = 1,n do
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+1, rnn_size)(h_old)) -- prev_c[L]
    table.insert(inputs, nn.Narrow(2, 2*(L-1)*rnn_size+rnn_size+1, rnn_size)(h_old)) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, noutput)(top_h)

 
	local h_new=nn.JoinTable(1,1)(outputs);
	local outs=proj;

  return nn.gModule({h_old,input},{h_new,outs})
end

return LSTM