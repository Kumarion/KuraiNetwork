# Kurai Neural Network
KuraiNetwork is a neural network library written in Luau. <br />

Note: This library is still in development and is not ready for production use. <br />
Note: I plan on scaling up the network to support more layers and more nodes per layer. I also plan on adding a genetic algorithm to train the network.

Only supports backpropagation for now.

## Usage
```lua
local KuraiNetwork = require(game:GetService("ReplicatedStorage"):WaitForChild("KuraiNetwork"));

-- Create a network with 2 inputs, 2 hidden nodes, and 1 output
local network = KuraiNetwork.new({
    inputSize = 2,
    hiddenSize = 2,
    outputSize = 1,
})

-- Create a learning set
local learningSet = {
    { { 0, 0 }, { 0 } },
    { { 0, 1 }, { 1 } },
    { { 1, 0 }, { 1 } },
    { { 1, 1 }, { 0 } },
}

-- Train the network
for i = 1, 10000 do
    local data = learningSet[math.random(1, #learningSet)];
    network:train(data[1], data[2], 0.1);
end

-- Test the network
for i = 1, 10 do
    local data = learningSet[math.random(1, #learningSet)];
    local input = data[1];
    local output = data[2];
    local prediction = network:predict(input);

    -- Print the input, output, and prediction
    print(string.format("Input: %s, Output: %s, Prediction: %s", input, output, prediction[1]));
end
```