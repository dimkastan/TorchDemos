----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'nn'
require 'optim'
require 'image'
require 'cunn'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-learningRateDecay', 1e-7, 'learning rate at t=0')
cmd:option('-batchSize', 200, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-cuda', false, 'use cuda')
cmd:option('-epochs', 500, 'number of epochs')
cmd:text()
opt = cmd:parse(arg)

print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'airplane', 'automobile', 'bird', 'cat',
     'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

   model = nn.Sequential()

if opt.network == '' then
   -- define model to train


   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolution(3,16, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolution(16,256, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(256*5*5))
      model:add(nn.Linear(256*5*5, 128))
      model:add(nn.Tanh())
      model:add(nn.Linear(128,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32, 1*32*32))
      model:add(nn.Tanh())
      model:add(nn.Linear(1*32*32, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()


if(opt.cuda) then
model:cuda()
criterion:cuda()
end
model:training()
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> using model:')
print(model)





----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   trsize = 50000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
end

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
    www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
    tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   indata = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.indata[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
-- starting from 1
trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.indata = trainData.indata[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
trainData.indata = trainData.indata:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
    local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
   -- normalize y ly:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
    local rgb = testData.data[i]
    local yuv = image.rgb2yuv(rgb)
   -- normalize y ly:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

-- display function
function display(input, model)


   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter % 10 == 0 then
      if opt.model == 'convnet' then
         local s=model:get(1).weight:size()
         win_w1 = image.display{
            image=model:get(1).weight:resize(s[1]*s[2],s[3],s[4]), zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='stage 1: weights', padding=1
         }
         s=model:get(4).weight:size()
         win_w2 = image.display{

            image=model:get(4).weight:resize(s[1]*s[2],s[3],s[4]), zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='stage 2: weights', padding=1
         }

         s=model.modules[4].output:size()

         win_w3 = image.display{

            image=model.modules[4].output:resize(s[1],s[2],s[3]), zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w3, legend='stage 4: responces', padding=1
         }

      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{
            image=W1, zoom=0.5, min=-1, max=1,
            win=win_w1, legend='W1 weights'
         }
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{
            image=W2, zoom=0.5, min=-1, max=1,
            win=win_w2, legend='W2 weights'
         }
      end
   end
   iter = iter + 1

end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   --  vars
    local time = sys.clock()
    local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
--       inputs = {}
      local  inputs=torch.FloatTensor(opt.batchSize,3,32,32)
      local  inputsrgb=torch.FloatTensor(opt.batchSize,3,32,32)
      local  targets = torch.FloatTensor(opt.batchSize)-- {}
      local  cntr=0;
      -- cst to float
       dataset.labels=dataset.labels:float();
       dataset.data=dataset.data:float();
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         cntr=cntr+1
         inputs[cntr]    =dataset.data[i]
         inputsrgb[cntr] =dataset.indata[i]
         targets[cntr]   =dataset.labels[i];
      end

      -- create closure to evaluate f(X) and df/dX
       feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local  f = 0
         if(opt.cuda) then
            targets=targets:cuda()
            inputs=inputs:cuda()
         end
         -- evaluate function for complete mini batch
         for i = 1,inputs:size(1) do
            -- estimate f

             local output = model:forward(inputs[i])

            local  err = criterion:forward(output, targets[i] )
             f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])

            -- visualize?
            -- if opt.visualize then
            if opt.visualize then
                if(opt.cuda) then
                    -- local temp=inputs:clone():float();
                    local mdl=model:clone()
                    mdl=mdl:float()

                    display(inputsrgb[{{i},{},{},{}}],mdl)
               end
            end

         end



         -- normalize gradients and f(X)
         gradParameters:div(inputs:size(1))
         f = f/inputs:size(1)
         trainError = trainError + f
         --print("train error= ",f)

         -- return f and df/dX
         return f,gradParameters
      end



      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         configParam = configParam or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, configParam)

      elseif opt.optimization == 'LBFGS' then
         configParam = configParam or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, configParam)

      elseif opt.optimization == 'SGD' then
         configParam = configParam or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = opt.learningRateDecay}
         optim.sgd(feval, parameters, configParam)

         print(configParam)

      elseif opt.optimization == 'ASGD' then
         configParam = configParam or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, configParam)

      else
         error('unknown optimization method')
      end
   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
    local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- save/log current net
local filename = paths.concat(opt.save, 'cifar.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)

   -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end
   local input= torch.FloatTensor(1,32,32)
   local target = torch.FloatTensor(1)
   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      input = dataset.data[t]:float()
      target[1] = dataset.labels[t]

      if(opt.cuda) then
         input  = input:cuda()
         target = target:cuda()
      end


      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target[1])

      -- compute error
      err = criterion:forward(pred, target)
      testError = testError + err
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end


----------------------------------------------------------------------
-- and train!
--
for epoch=1,opt.epochs do
    print("Epoch # ",epoch)
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)
   -- update logger
    accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
    errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

    -- plot logger
    accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
    errLogger:style{['% train error']    = '-', ['% test error']    = '-'}

    accLogger:plot()
    errLogger:plot()

    opt.learningRate=opt.learningRate - opt.learningRateDecay;

end
