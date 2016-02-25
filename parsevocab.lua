require 'torch'

--get vocab
vocabFile = torch.DiskFile('vocab_alpha_space.asc', 'r')
vocab = vocabFile:readObject()

cuArray = torch.eye(57)

function getOneHotChar(c)
    local index = vocab[c]
    local oneHotChar = cuArray[index]
    return oneHotChar
end

function getOneHotStr(s)
    local oneHotStr = nil
    for c in s:gmatch"." do
        
        res = c:gsub('[^a-zA-Z .!?]','')
        if res ~= '' then
            inputchar = c
        else
            inputchar = '0'
        end

        if not oneHotStr then
            oneHotStr = getOneHotChar(inputchar)
        else
            oneHotStr = torch.cat(oneHotStr, getOneHotChar(inputchar), 2)
        end
    end
    return oneHotStr:clone()
end

function getOneHotStrs(strs)

-- will be given as an array of strs to be converted into one 
-- hot array of arrays

    maxCharLen = 0

    for i = 1, #strs do
        charLen = #strs[i]
        if charLen > maxCharLen then
            maxCharLen = charLen
        end 
    end
    
    --allOneHot = torch.zeros(83, maxCharLen, #strs)
    allOneHot = torch.zeros(#strs, maxCharLen, 57)
    
    for i = 1, #strs do
        strLen = #(strs[i])
        charRemain = maxCharLen - strLen
        oneHot = getOneHotStr(strs[i])
        if charRemain > 0 then
            zeroOneHotVectors = torch.zeros(57, charRemain)
            finalOneHot = torch.cat(oneHot, zeroOneHotVectors,2)
            allOneHot[{{i},{},{}}] = finalOneHot:t()
        else
            allOneHot[{{i},{},{}}] = oneHot:t()
        end
    end 

    return allOneHot
end
