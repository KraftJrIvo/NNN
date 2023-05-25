#include <iostream>
#include "..\NNN\NeuralNet.hpp"

using namespace nnn;

int main()
{
	NNDataset<float, 2, 1> op_and = {
		{
			{0.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f}
		},
		{
			{0.0f},
			{0.0f},
			{0.0f},
			{1.0f}
		}
	};

	NNDataset<float, 2, 1> op_or = {
		{
			{0.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f}
		},
		{
			{0.0f},
			{1.0f},
			{1.0f},
			{1.0f}
		}
	};

	NNDataset<float, 2, 1> op_xor = {
		{
			{0.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f}
		},
		{
			{0.0f},
			{1.0f},
			{1.0f},
			{0.0f}
		}
	};

	NNDesc desc{
		{
			{2, ActivationFunctionType::NONE}, 
			{2, ActivationFunctionType::SIGMOID}, // this layer is only needed for xor
			{1, ActivationFunctionType::SIGMOID}
		}, 
		BackPropagationMethod::REGULAR,
		LossFunctionType::L2
	};

	NNDataset<float, 2, 1>& dataset = op_xor;

	NeuralNet<float, 2, 1> nn(desc);

	nn.initialize(-1.0f, 1.0f);

	nn.test(dataset);

	std::cout << "training...\n";

	nn.train(dataset, 1000, 4, 10.0f);

	nn.test(dataset);

	std::cout << "done.\n";

	system("pause");
}