require 'paths'
require 'nn'


-- Very simple 

-- Author: Barry Kui
-- Date: 21-Jul-16

-- We do 5 steps left to do in training our first torch neural network
-- 1. Load and normalize data
-- 2. Define Neural Network
-- 3. Define Loss function
-- 4. Train network on training data
-- 5. Test network on test data.



--------------------------------------
-- 1. Load and normalize data  
--------------------------------------
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
print(#trainset.data)

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end
print(trainset:size())

redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}

print("num of redChannel:",#redChannel)


-- doing mean-subtraction and standard-deviation based scaling

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
-- OK! our training data is now normalized and ready to be used.




--------------------------------------
-- 2. Define Neural Network
--------------------------------------
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems


--------------------------------------
-- 3. Define Loss function  
--------------------------------------
criterion = nn.ClassNLLCriterion()


--------------------------------------
-- 4. Train network on training data 
--------------------------------------
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)


--------------------------------------
-- 5. Test network on test data.
--------------------------------------
-- We have trained the network for 2 passes over the training dataset.
-- But we need to check if the network has learnt anything at all.
-- We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.
-- Okay, first step. Let us display an image from the test set to get familiar.

print(classes[testset.label[100]])
-- Now that we are done with that, let us normalize the test data with the mean and standard-deviation from the training data.

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- for fun, print the mean and standard-deviation of example-100
-- horse = testset.data[100]
-- print(horse:mean(), horse:std())
print(classes[testset.label[100]])
-- itorch.image(testset.data[100])
predicted = net:forward(testset.data[100])

-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())


-- To make it clearer, let us tag each probability with it's class-name:
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

-- Alright, fine. One single example sucked, but how many in total seem to be correct over the test set?

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(correct, 100*correct/10000 .. ' % ')


-- That looks waaay better than chance, which is 10% accuracy (randomly picking a class out of 10 classes). Seems like the network learnt something.
-- Hmmm, what are the classes that performed well, and the classes that did not perform well:


class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end

-- Okay, so what next? How do we run this neural network on GPUs?
-- just run "th train_gpu.lua" 
