--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'nn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local ptb = require('data')

local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=1.3,
                embed_size=50,
                rnn_size=100,
                dropout=.5,
                init_weight=0.1,
                lr=1,
                vocab_size=130001,
                tag_size=46,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

local function transfer_data(x)
  return x--:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[-1] = LookupTable(params.vocab_size,
                                                    params.embed_size)(x)}
  i[0]                   = nn.Linear(params.embed_size, params.rnn_size)(i[-1])
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.tag_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(pred), nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  ptb.load_weights(i[-1].data.module.weight)
  return transfer_data(module)
end

local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  model.out = {}
  for j = 0, 2 * params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  for j = 0, 2 * params.seq_length do
      model.out[j] = transfer_data(torch.zeros(params.batch_size, params.tag_size))
  end
  model.dout = transfer_data(torch.zeros(params.batch_size, params.tag_size))
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, 2 * params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(2 * params.seq_length))
end

local function reset_state(state)
  state.correct = 0
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    model.err[i], model.out[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  state.pos = state.pos - params.seq_length
  for i = params.seq_length + 1, 2 * params.seq_length do
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    model.err[i], model.out[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    local _,o = model.out[i]:max(2)
    o = o:double()
    local correct = y:eq(o:reshape(y:size()))
    state.correct = state.correct + correct:sum()
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[2 * params.seq_length])
  return model.err:mean()
end

local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = 2 * params.seq_length, params.seq_length + 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.dout, model.ds})
    g_replace_table(model.ds, tmp[3])
    --g_replace_table(model.dout, tmp[2])
    --cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.dout, model.ds})
    g_replace_table(model.ds, tmp[3])
    --g_replace_table(model.dout, tmp[2])
    --cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  print("Validation set accuracy : " .. g_f3(state_valid.correct / (state_valid.pos-1) / params.batch_size))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  local tcor = 0
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.label[i]
    local s = model.s[i - 1]
    perp_tmp, output, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]

    local _,o = output:max(2)
    o = o:double()
    local correct = y:eq(o:reshape(y:size()))
    tcor = tcor + correct:sum()

    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  print("Test set accuracy : " .. g_f3(tcor / len / params.batch_size))
  g_enable_dropout(model.rnns)
end

local function main()
  --g_init_gpu(arg)
  local d,l = ptb.traindataset(params.batch_size)
  state_train = {data=d, label=l}
  d,l = ptb.validdataset(params.batch_size)
  state_valid =  {data=d, label=l}
  d,l = ptb.testdataset(params.batch_size)
  state_test =  {data=d, label=l}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    local tot = (state_train.pos-1)*params.batch_size
    local cor = state_train.correct
    local wps = torch.floor(total_cases / torch.toc(start_time))
    local since_beginning = g_d(torch.toc(beginning_time) / 60)
    print('epoch = ' .. g_f3(epoch) ..
          ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
          ', wps = ' .. wps ..
          ', dw:norm() = ' .. g_f3(model.norm_dw) ..
          ', correct = ' .. cor .. '/' .. tot .. '  ' .. g_f3(cor / tot) ..
          ', lr = ' ..  g_f3(params.lr) ..
          ', since beginning = ' .. since_beginning .. ' mins.')
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      --cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  print("Training is over.")
end

main()
