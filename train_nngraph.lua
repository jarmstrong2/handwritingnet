require 'getBatch'
--params:uniform(-0.08, 0.08)
sampleSize = opt.batchSize
numberOfPasses = opt.numPasses

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
initstate_h1_c = torch.zeros(sampleSize, opt.hiddenSize):cuda()
initstate_h1_h = initstate_h1_c:clone()
initstate_h2_c = initstate_h1_c:clone()
initstate_h2_h = initstate_h1_c:clone()
initstate_h3_c = initstate_h1_c:clone()
initstate_h3_h = initstate_h1_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
dfinalstate_h1_c = initstate_h1_c:clone()
dfinalstate_h1_h = initstate_h1_c:clone()
dfinalstate_h2_c = initstate_h1_c:clone()
dfinalstate_h2_h = initstate_h1_c:clone()
dfinalstate_h3_c = initstate_h1_c:clone()
dfinalstate_h3_h = initstate_h1_c:clone()
initkappa = torch.randn(sampleSize,10)
dinitkappa = torch.zeros(sampleSize,10)

count = 1

batchCount = nil

function getInitW(cuMat)
    return cuMat[{{},{1},{}}]:squeeze(2)
end

function makecov(std, rho)
    covmat = torch.Tensor(2,2)
    covmat[{{1},{1}}] = torch.pow(std[{{1},{1}}], 2)
    covmat[{{1},{2}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{1}}] = torch.cmul(torch.cmul(std[{{1},{1}}], std[{{1},{2}}]), rho[{{1},{2}}])
    covmat[{{2},{2}}] = torch.pow(std[{{1},{2}}], 2)
    return covmat
end

function getSample(sampleSize, yOutput)
    sampX = torch.zeros(sampleSize, 3)
    for i=1,sampleSize do
        currentY = yOutput[{{i},{}}]
        x_1, x_2, x_3 = _getSample(currentY)
        sampX[{{i},{1}}] = x_1
        sampX[{{i},{2}}] = x_2
        sampX[{{i},{3}}] = x_3
    end
    return sampX:cuda()
end

function _getSample(input)
    e_t = input[{{},{1}}]
    pi_t = input[{{},{2,21}}]
    mu_1_t = input[{{},{22,41}}]
    mu_2_t = input[{{},{42,61}}]
    sigma_1_t = input[{{},{62,81}}]
    sigma_2_t = input[{{},{82,101}}]
    rho_t = input[{{},{102,121}}]
    
    x_3 = torch.Tensor(1)
    x_3 = (x_3:bernoulli(e_t:squeeze())):squeeze()
    
    chosen_pi = torch.multinomial(pi_t:double(), 1):squeeze()

    curstd = torch.Tensor({{sigma_1_t[{{},{chosen_pi}}]:squeeze(), sigma_2_t[{{},{chosen_pi}}]:squeeze()}})
    curcor = torch.Tensor({{1, rho_t[{{},{chosen_pi}}]:squeeze()}})
    curcovmat = makecov(curstd, curcor)
    curmean = torch.Tensor({{mu_1_t[{{},{chosen_pi}}]:squeeze(), mu_2_t[{{},{chosen_pi}}]:squeeze()}})
    sample = distributions.mvn.rnd(curmean, curcovmat)
    x_1 = sample[1]
    x_2 = sample[2]
    return x_1, x_2, x_3
end

function getValLoss()
    local valnumberOfPasses = opt.numPasses
    local valcount = 1
    local valsampleSize = opt.batchSize
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, valnumberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, ymaskMat, wmaskMat, cmaskMat, elementCount, 
        valcount = getBatch(valcount, valhandwritingdata, valsampleSize)
        ------------------------------------------------------------

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        
        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- forward
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{},{t+1}}]:squeeze(3)

            -- model 
            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]}))
       
            -- criterion 
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss        
        end

        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    return loss
end

function schedSampBool() 
    k = 0.9
    i = batchCount/80.0
    e_i = k^i
    -- if we get 1 then don't sample, if 0 then do sample
    randvar = torch.Tensor(1)
    result = randvar:bernoulli(e_i):squeeze()
    return result  
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local loss = 0
    local elems = 0
    
    -- add for loop to increase mini-batch size
    for i=1, numberOfPasses do

        --------------------- get mini-batch -----------------------
        maxLen, strs, inputMat, cuMat, ymaskMat, wmaskMat, cmaskMat, elementCount, 
        count = getBatch(count, handwritingdata, sampleSize)
        ------------------------------------------------------------

        if maxLen > MAXLEN then
            maxLen = MAXLEN
        end

        -- initialize window to first char in all elements of the batch
        local w = {[0]=getInitW(cuMat:cuda())}

        local lstm_c_h1 = {[0]=initstate_h1_c} -- internal cell states of LSTM
        local lstm_h_h1 = {[0]=initstate_h1_h} -- output values of LSTM
        local lstm_c_h2 = {[0]=initstate_h2_c} -- internal cell states of LSTM
        local lstm_h_h2 = {[0]=initstate_h2_h} -- output values of LSTM
        local lstm_c_h3 = {[0]=initstate_h3_c} -- internal cell states of LSTM
        local lstm_h_h3 = {[0]=initstate_h3_h} -- output values of LSTM
        
        local kappa_prev = {[0]=torch.zeros(sampleSize,10):cuda()}
        
        local output_h1_w = {}
        local input_h3_y = {}
        local output_h3_y = {}
        local output_y = {}
        
        -- forward
        
        --print('forward')
        
        for t = 1, maxLen - 1 do
            local x_in = inputMat[{{},{},{t}}]:squeeze(3)
            local x_target = inputMat[{{},{},{t+1}}]:squeeze(3)
       
            -- Using Scheduled Sampling
            -- if returns 1 then don't sample, o.w. do
            --sampleBool = schedSampBool()

            --if sampleBool == 0 and t ~= 1 then
            --    x_in = getSample(sampleSize, output_y[t-1])
            --end

            -- model 
            output_y[t], kappa_prev[t], w[t], _, lstm_c_h1[t], lstm_h_h1[t],
            lstm_c_h2[t], lstm_h_h2[t], lstm_c_h3[t], lstm_h_h3[t]
        = unpack(clones.rnn_core[t]:forward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]}))
       
            -- criterion 
            --print(cmaskMat:clone():sum(3):squeeze())
            clones.criterion[t]:setmask(cmaskMat[{{},{},{t}}]:cuda())
            loss = clones.criterion[t]:forward(output_y[t], x_target:cuda()) + loss
            --print('inner loop ',loss)        
        end
        --print('current pass ',loss)        
        elems = (elementCount - sampleSize) + elems
        
        -- backward
        
        --print('backward')
        
        local dlstm_c_h1 = dfinalstate_h1_c
        local dlstm_h_h1 = dfinalstate_h1_h
        local dlstm_c_h2 = dfinalstate_h2_c
        local dlstm_h_h2 = dfinalstate_h2_h
        local dlstm_c_h3 = dfinalstate_h3_c
        local dlstm_h_h3 = dfinalstate_h3_h
        
        local dh1_w = torch.zeros(sampleSize, 57):cuda()
        local dkappa = torch.zeros(sampleSize, 10):cuda()
        
        for t = maxLen - 1, 1, -1 do
        
            local x_in = inputMat[{{},{},{t}}]:squeeze()
            local x_target = inputMat[{{},{},{t+1}}]:squeeze()
            
            -- criterion
            local grad_crit = clones.criterion[t]:backward(output_y[t], x_target:cuda())
	    grad_crit:clamp(-100,100)            

            -- model
            _x, _c, dkappa, dh1_w, dlstm_c_h1, dlstm_h_h1,
            dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 = unpack(clones.rnn_core[t]:backward({x_in:cuda(), cuMat:cuda(), 
                 kappa_prev[t-1], w[t-1], lstm_c_h1[t-1], lstm_h_h1[t-1],
                 lstm_c_h2[t-1], lstm_h_h2[t-1], lstm_c_h3[t-1], lstm_h_h3[t-1]},
                 {grad_crit, dkappa, dh1_w, _, dlstm_c_h1, dlstm_h_h1, 
                  dlstm_c_h2, dlstm_h_h2, dlstm_c_h3, dlstm_h_h3 }))

            -- clip gradients
            dh1_w:clamp(-10,10)
            dlstm_c_h1:clamp(-10,10)
            dlstm_h_h1:clamp(-10,10)
            dlstm_c_h2:clamp(-10,10)
            dlstm_h_h2:clamp(-10,10)
            dlstm_c_h3:clamp(-10,10)
            dlstm_h_h3:clamp(-10,10)
        end
    
        dh2_w = nil
        dh2_h1 = nil
        dh3_w = nil
        dh3_h2 = nil
        maxLen = nil
        strs = nil
        inputMat = nil 
        maskMat = nil
        cuMat = nil
        w = nil
        lstm_c_h1 = nil -- internal cell states of LSTM
        lstm_h_h1 = nil -- output values of LSTM
        lstm_c_h2 = nil -- internal cell states of LSTM
        lstm_h_h2 = nil -- output values of LSTM
        lstm_c_h3 = nil -- internal cell states of LSTM
        lstm_h_h3 = nil -- output values of LSTM
        dlstm_c_h1 = nil -- internal cell states of LSTM
        dlstm_h_h1 = nil -- internal cell states of LSTM
        dlstm_c_h2 = nil -- internal cell states of LSTM
        dlstm_h_h2 = nil -- internal cell states of LSTM
        dlstm_c_h3 = nil -- internal cell states of LSTM
        dlstm_h_h3 = nil -- internal cell states of LSTM
        dkappaNext = nil
        dh1_w_next = nil
        kappa_prev = nil
        output_h1_w = nil
        input_h3_y = nil
        output_h3_y = nil
        output_y = nil
        collectgarbage()
    end
    
    grad_params:div(numberOfPasses)
    
    -- clip gradient element-wise
    grad_params:clamp(-10, 10)
    
    
    return loss, grad_params
end

losses = {} 
vallosses = {}
local optim_state = {learningRate = opt.lr, alpha = 0.95, epsilon = 1e-6}
local iterations = 8000
local minValLoss = 1/0
for i = 1, iterations do
    batchCount = i

    local _, loss = optim.adam(feval, params, optim_state)

    print(string.format("update param, loss = %6.8f, gradnorm = %6.4e", loss[1], grad_params:clone():norm()))
    if i % 20 == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
        valLoss = getValLoss()
        vallosses[#vallosses + 1] = valLoss
        print(string.format("validation loss = %6.8f", valLoss))
        if minValLoss > valLoss then
            minValLoss = valLoss
            torch.save("alexnet.t7", model)
            print("------- Model Saved --------")
        end
        losses[#losses + 1] = loss[1]
        torch.save("losses.t7", losses)
        torch.save("vallosses.t7", vallosses)
    end
end
