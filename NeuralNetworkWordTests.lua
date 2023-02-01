-- test with words  
-- soon
local KuraiNetwork = require(game.ReplicatedStorage.NN)
local network = KuraiNetwork.new(2, 2, 1);

local trainingSet = {
    --  Set,    Target
    { { "a", "b" }, { "c" } },
    { { "b", "c" }, { "d" } },
    { { "c", "d" }, { "e" } },
    { { "d", "e" }, { "f" } },
};

local learningRate = 0.1

for i = 1, 1000000 do
    for j = 1, #trainingSet do
        local input = trainingSet[j][1];
        local target = trainingSet[j][2];
        
        network:train(input, target, learningRate);
    end;
end;

local testSet = {
    --  Set,    Target
    { { "a", "b" }, { "c" } },
    { { "b", "c" }, { "d" } },
    { { "c", "d" }, { "e" } },
    { { "d", "e" }, { "f" } },
};

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