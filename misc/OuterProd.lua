--[[
  Input: a table of 2 tensors.
  Output: the outer product of the vectors.
--]]

-- modified from source: https://github.com/kaishengtai/torch-ntm/blob/master/layers/OuterProd.lua

require 'nn'

local OuterProd, parent = torch.class('nn.OuterProd', 'nn.Module')

function OuterProd:__init()
  parent.__init(self)
  self.gradInput = {}
end

function OuterProd:updateOutput(input)
  local order = #input
  -- self.order = order
  
  if order == 2 then
    batch_size = input[1]:size()[1]
    input1_size = input[1]:size()[2]
    input2_size = input[2]:size()[2]

    -- local output = torch.zeros(batch_size, input1_size, input2_size)
    -- output = output:cuda()

    self.output = torch.zeros(batch_size, input1_size, input2_size):typeAs(input[1])

    for i = 1, batch_size do
      self.output[i] = torch.ger(input[1][i], input[2][i])
    end
    -- self.size = self.output:size()
    -- self.output = output
  else
    error('outer products of order higher than 2 unsupported')
  end

  return self.output
end

function OuterProd:updateGradInput(input, gradOutput)
  local order = #input
  for i = 1, order do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i])
  end

  if order == 2 then
    batch_size = input[1]:size()[1]

    for i = 1, batch_size do
      self.gradInput[1][i]:copy(gradOutput[i] * input[2][i])
      self.gradInput[2][i]:copy(gradOutput[i]:t() * input[1][i])
    end
  else
    error('outer products of order higher than 2 unsupported')
  end
  return self.gradInput
end