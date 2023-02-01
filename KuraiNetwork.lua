-- Kurai neural network class
local KuraiNetwork = { };

function KuraiNetwork.new(inputSize, hiddenSize, outputSize)
    local self = { };

    self.inputSize = inputSize;
    self.hiddenSize = hiddenSize;
    self.outputSize = outputSize;

    self.weights1 = KuraiNetwork.generateWeights(self.inputSize, self.hiddenSize);
    self.bias1 = KuraiNetwork.generateBias(1, self.hiddenSize);

    self.weights2 = KuraiNetwork.generateWeights(self.hiddenSize, self.outputSize);
    self.bias2 = KuraiNetwork.generateBias(1, self.outputSize);

    return setmetatable(self, { __index = KuraiNetwork });
end

function KuraiNetwork.generateWeights(inputSize, outputSize)
	local weights = { };
	for i = 1, inputSize do
		weights[i] = { };
		for j = 1, outputSize do
			weights[i][j] = math.random();
		end;
	end;
	return weights;
end

function KuraiNetwork.generateBias(inputSize, outputSize)
	local bias = { };
	for i = 1, inputSize do
		bias[i] = { };
		for j = 1, outputSize do
			bias[i][j] = math.random();
		end;
	end;
	return bias;
end

function KuraiNetwork:sigmoid(x)
	return 1 / (1 + math.exp(-x));
end

-- Feed forward
function KuraiNetwork:forward(input)
    local hiddenLayer = { };
    for i = 1, self.hiddenSize do
        local sum = self.bias1[1][i];
        for j = 1, self.inputSize do
            sum = sum + (input[j] * self.weights1[j][i]);
        end;
        hiddenLayer[i] = self:sigmoid(sum);
    end;

    local outputLayer = { };
    for i = 1, self.outputSize do
        local sum = self.bias2[1][i];
        for j = 1, self.hiddenSize do
            sum = sum + (hiddenLayer[j] * self.weights2[j][i]);
        end;
        outputLayer[i] = self:sigmoid(sum);
    end;

    return outputLayer;
end

-- Backpropagation
function KuraiNetwork:backprop(input, target, learningRate)
    local hiddenLayer = { };
    for i = 1, self.hiddenSize do
        local sum = self.bias1[1][i];
        for j = 1, self.inputSize do
            sum = sum + (input[j] * self.weights1[j][i]);
        end;
        hiddenLayer[i] = self:sigmoid(sum);
    end;

    local outputLayer = { };
    for i = 1, self.outputSize do
        local sum = self.bias2[1][i];
        for j = 1, self.hiddenSize do
            sum = sum + (hiddenLayer[j] * self.weights2[j][i]);
        end;
        outputLayer[i] = self:sigmoid(sum);
    end;

    local outputError = { };
    for i = 1, self.outputSize do
        outputError[i] = target[i] - outputLayer[i];
    end;

    local hiddenError = { };
    for i = 1, self.hiddenSize do
        local sum = 0;
        for j = 1, self.outputSize do
            sum = sum + (outputError[j] * self.weights2[i][j]);
        end;
        hiddenError[i] = sum;
    end;

    for i = 1, self.hiddenSize do
        for j = 1, self.outputSize do
            self.weights2[i][j] = self.weights2[i][j] + (learningRate * outputError[j] * outputLayer[j] * (1 - outputLayer[j]) * hiddenLayer[i]);
        end;
    end;

    for i = 1, self.inputSize do
        for j = 1, self.hiddenSize do
            self.weights1[i][j] = self.weights1[i][j] + (learningRate * hiddenError[j] * hiddenLayer[j] * (1 - hiddenLayer[j]) * input[i]);
        end;
    end;
end

-- Train
function KuraiNetwork:train(input, target, learningRate)
    self:backprop(input, target, learningRate);
end

-- Predict
function KuraiNetwork:predict(input)
    local output = self:forward(input);
    return output;
end

-- test train
function KuraiNetwork:trainingWheels()
    local set = {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 1 } },
        { { 1, 0 }, { 1 } },
        { { 1, 1 }, { 0 } }
    };

    for i = 1, 50 do
        for j = 1, #set do
            local input = set[j][1];
            local target = set[j][2];
            
            self:train(input, target, 0.1);
        end;
    end;
end

return KuraiNetwork;