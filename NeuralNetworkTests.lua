local KuraiNetwork = require(game.ReplicatedStorage.NN)

-- Usage of KuraiNetwork

-- Create a new KuraiNetwork object
-- The first argument is the number of inputs
-- The second argument is the number of hidden nodes
-- The third argument is the number of outputs
local network = KuraiNetwork.new(2, 2, 1);

-- Create a training set
local trainingSet = {
	--  Set,    Target
	{ { 0, 0 }, { 0 } },
	{ { 0, 1 }, { 1 } },
	{ { 1, 0 }, { 1 } },
	{ { 1, 1 }, { 0 } },
};

-- Train the network
-- network:train(trainingSet, 1000000, 0.1);
-- network:trainingWheels()
local learningRate = 0.1
for i = 1, 1000000 do
    for j = 1, #trainingSet do
        local input = trainingSet[j][1];
        local target = trainingSet[j][2];
        
        network:train(input, target, learningRate);
    end;
end;

-- Test the network
-- A good test set is one that the network has not seen before
-- Not the same as the training set
local testSet = {
	--  Set,    Target
	{ { 0, 0 }, { 0 } },
	{ { 0, 1 }, { 1 } },
	{ { 1, 0 }, { 1 } },
	{ { 1, 1 }, { 0 } },
};

-- If all goes well, the network should be able to predict the target
for i = 1, #testSet do
    local input = testSet[i][1];
    local target = testSet[i][2];
    local output = network:predict(input);

    print("Test #" .. i);
    print("Input: " .. input[1] .. ", " .. input[2]);
    print("Target: " .. target[1]);
    print("AI Guess:", output);
    print("-----------------------");
end