#include <iostream>
#include "..\NNN\NeuralNet.hpp"

using namespace nnn;

int main()
{
	NNDataset<float, 1, 1> line = {
		{
			{0.0f},
			{1.0f},
			{2.0f},
			{3.0f},
			{43.0f},
			{11.0f},
			{12.0f},
			{13.0f},
		},
		{
			{1.0f},
			{3.0f},
			{5.0f},
			{7.0f},
			{87.0f},
			{23.0f},
			{25.0f},
			{27.0f},
		}
	};

	NNDesc desc{
		{
			{1, ActivationFunctionType::NONE},
			{1, ActivationFunctionType::NONE}
		},
		BackPropagationMethod::REGULAR,
		LossFunctionType::L2
	};

	NeuralNet<float, 1, 1> nn(desc);

	nn.initialize(0.0f, 1.0f);

	nn.test(line);

	std::cout << "training...\n";

	nn.train(line, 1000, 8, 0.00005f, true);

	nn.test(line);

	std::cout << "done.\n";

	system("pause");
}