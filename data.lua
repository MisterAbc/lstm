--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local tablex = require('pl.tablex')
local file = require('pl.file')

local ptb_path = "./data/"
local dname = "./data/sentences.txt"
local lname = "./data/tags.txt"
local vmapname = "./embeddings/words.lst"
local lmapname = "./data/tags.lst"
local wname = "./embeddings/embeddings.txt"

local vocab_map = nil
local label_map = nil

local dataset = nil
local labelset = nil

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
  local s = x_inp:size(1)
  local x = torch.zeros(torch.floor(s / batch_size), batch_size)
  for i = 1, batch_size do
    local start = torch.round((i - 1) * s / batch_size) + 1
    local finish = start + x:size(1) - 1
    x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
  end
  return x
end

local function load_data()
  if vocab_map == nil then
    local data = file.read(vmapname)
    data = stringx.split(data)
    local s = #data
    data = tablex.index_map(data)
    vocab_map = data
    -- +1 for end of sentence marker
    vocab_map.size = function() return s+1 end
  end

  if label_map == nil then
    local data = file.read(lmapname)
    data = stringx.split(data)
    local s = #data
    data = tablex.index_map(data)
    label_map = data
    -- +1 for end of sentence marker
    label_map.size = function() return s+1 end
  end

  if dataset == nil then
    local data = file.read(dname)
    data = stringx.replace(data, '\n', ' ' .. vocab_map.size() .. ' ')
    data = stringx.split(data)
    print(string.format("Loading %s, size of data = %d", dname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
      x[i] = tonumber(data[i])
    end
    dataset = x:clone()
  end

  if labelset == nil then
    local data = file.read(lname)
    data = stringx.replace(data, '\n', ' ' .. label_map.size() .. ' ')
    data = stringx.split(data)
    print(string.format("Loading %s, size of labels = %d", lname, #data))
    x = torch.zeros(#data)
    for i = 1, #data do
      x[i] = tonumber(data[i])
    end
    labelset = x
  end

  return dataset, labelset
end

local function load_weights(tensor)
  local data = file.read(wname)
  data = stringx.lines(data)
  local i = 1
  for line in data do
    local nums = stringx.split(line)
    nums = torch.Tensor(tablex.map(tonumber, nums))
    if nums:size(1) ~= tensor[i]:size(1) then
      error("Lookup table embedding size is not right")
    end
    tensor[i]:copy(nums)
    i = i+1
    if i > tensor:size(1) then
      error("Lookup table doesn't have enough elements to load weights")
    end
  end
end

local function traindataset(batch_size)
  local x, y = load_data()
  local e = torch.floor( x:size(1)*3/5 )
  x = x[{{1, e}}]
  y = y[{{1, e}}]
  x = replicate(x, batch_size)
  y = replicate(y, batch_size)
  return x, y
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
  local x, y = load_data()
  local s = torch.floor ( x:size(1)*3/5+1 )
  local e = torch.floor ( x:size(1)*4/5+1 )
  x = x[{{s,e}}]
  y = y[{{s,e}}]
  x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
  y = y:resize(y:size(1), 1):expand(y:size(1), batch_size)
  return x, y
end

local function validdataset(batch_size)
  local x, y = load_data()
  local s = torch.floor ( x:size(1)*4/5+1 )
  local e = x:size(1)
  x = x[{{s,e}}]
  y = y[{{s,e}}]
  x = replicate(x, batch_size)
  y = replicate(y, batch_size)
  return x, y
end

return {traindataset=traindataset,
testdataset=testdataset,
validdataset=validdataset,
load_weights=load_weights}
