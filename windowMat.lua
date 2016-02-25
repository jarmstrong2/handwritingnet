require 'nn'

local Window, Parent = torch.class('nn.Window', 'nn.Module')

function Window:updateOutput(input)
    local input_h1, context, kappas_t_1, _ = unpack(input)
    self.kappas_t_1 = kappas_t_1

    self.cu = context
    self.cu_size = (#self.cu[{{1},{},{}}])[2]
    self.vocab_size = (#self.cu[{{1},{},{}}])[3]
    
    local input_cp = input_h1:clone()
    
    local hat_alphas_t = input_cp[{{},{1,10}}]
    local hat_betas_t = input_cp[{{},{11,20}}]
    local hat_kappas_t = input_cp[{{},{21,30}}]
    
    local alphas_t = torch.exp(hat_alphas_t):cuda()
    local betas_t = torch.exp(hat_betas_t):cuda()
    local kappas_t = self.kappas_t_1 + torch.exp(hat_kappas_t):cuda()
    
    local sampleSize = (#alphas_t)[1]
    
    local u_vector = torch.linspace(1, self.cu_size, self.cu_size):cuda()
    
    local u_expanded = u_vector:resize(1, 1, self.cu_size):expand(sampleSize, 10, self.cu_size)
    
    local kappas_t_expanded = kappas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    local betas_t_expanded = betas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    local alphas_t_expanded = alphas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    local calc = torch.pow(kappas_t_expanded - u_expanded, 2)
    
    calc:cmul(-betas_t_expanded)
    
    calc:exp()
    
    calc:cmul(alphas_t_expanded)
    
    local phi_t = torch.sum(calc, 2)
    
    local cu_resized = self.cu:clone()
    
    local output = torch.bmm(phi_t:cuda(), cu_resized:cuda()):squeeze(2)
    
    self.output = {output, kappas_t, phi_t}
    
    return self.output
end

function Window:updateGradInput(input, gradOutput)
    local input_h1, context, kappas_t_1, mask = unpack(input)
    local grad_output, d_kappas_t_plus_1 = unpack(gradOutput)
    
    local input_cp = input_h1:clone()
    
    local hat_alphas_t = input_cp[{{},{1,10}}]
    local hat_betas_t = input_cp[{{},{11,20}}]
    local hat_kappas_t = input_cp[{{},{21,30}}]
     
    local alphas_t = torch.exp(hat_alphas_t):cuda()
    local betas_t = torch.exp(hat_betas_t):cuda()
    local kappas_t = self.kappas_t_1 + torch.exp(hat_kappas_t):cuda()
    
    local sampleSize = (#alphas_t)[1]
   
    -- calculate epsilon(k,t,u)
    
    local gradOutput_expanded = grad_output:clone():resize(sampleSize, 1, self.vocab_size)
    :expand(sampleSize, self.cu_size, self.vocab_size)
    
    local cu_resized = self.cu:clone()
    
    local calc = torch.cmul(gradOutput_expanded:cuda(), cu_resized:cuda())
    calc = calc:sum(3):squeeze(3)
    
    local gradSum = calc:clone():resize(sampleSize, 1, self.cu_size)
    :expand(sampleSize, 10, self.cu_size)
    
    local u_vector = torch.linspace(1, self.cu_size, self.cu_size):cuda()
    
    local u_expanded = u_vector:resize(1, 1, self.cu_size):expand(sampleSize, 10, self.cu_size)
    
    local kappas_t_expanded = kappas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    local betas_t_expanded = betas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    local alphas_t_expanded = alphas_t:clone():resize(sampleSize, 10, 1):expand(sampleSize, 10, self.cu_size)
    
    calc = torch.pow(kappas_t_expanded - u_expanded, 2)
    
    calc:cmul(-betas_t_expanded)
    
    calc:exp()
    
    calc:cmul(alphas_t_expanded)


    local epsilon = torch.cmul(calc, gradSum)
    
    --compute dl_dalphas_hat 
    local dl_dalphas_hat = torch.sum(epsilon, 3):squeeze(3)
    
    --compute dl_dbetas_hat
    local dl_dbetas_hat = torch.pow(kappas_t_expanded - u_expanded, 2)
    dl_dbetas_hat:cmul(epsilon)
    dl_dbetas_hat = torch.sum(dl_dbetas_hat, 3):squeeze(3)
    dl_dbetas_hat:cmul(-betas_t)

    --compute dl_dkappas
    local dl_dkappas = torch.cmul(epsilon, u_expanded - kappas_t_expanded)
    dl_dkappas = torch.sum(dl_dkappas, 3):squeeze(3)
    dl_dkappas:cmul(betas_t)
    dl_dkappas:mul(2)
    dl_dkappas:add(d_kappas_t_plus_1)
    
    --compute dl_dkappas_hat
    local dl_dkappas_hat = torch.cmul(torch.exp(hat_kappas_t), dl_dkappas)
   
    local grad_input = torch.cat(dl_dalphas_hat:float(), torch.cat(dl_dbetas_hat:float(), dl_dkappas_hat:float(), 2), 2):squeeze()
    local grad_context = context:clone():zero()
   
    self.gradInput = {grad_input:cuda(), grad_context, dl_dkappas}
    
    return self.gradInput
    
end
