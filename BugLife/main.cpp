#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"
#include "World.hpp"
#include "Drawer.h"

using namespace nnn;

// in
// prv(boost d_ori a_bite a_lay) nrg d_nrg e_ori closeness r g b  (11)
// out
// vel d_ori a_bite a_lay

void prepareMinimalSurvivingSpecimen() {

	std::vector<SampleIn<float, 11>> ins;
	std::vector<SampleOut<float, 4>> outs;
	for (int i = 0; i < 10000; ++i) {
		SampleIn<float, 11> in;
		for (int j = 0; j < in.cols(); ++j)
			in[j] = BL_RAND_FLOAT;
		ins.push_back(in);
		float& energy = in[4];
		float& closeness = in[7];
		float& r = in[8]; float& g = in[9]; float& b = in[10];
		
		bool seesFood = (r > 0.5 || g > 0.5 || b > 0.5);
		float boost = seesFood ? 0.75f : 0.5f;
		float d_ori = (seesFood && closeness < 0.75f) ? 0.5f : 0.75f;
		float tryToBite = seesFood && (closeness > 0 && closeness <= 0.1);
		float tryToLay = (energy > 0.95) ? 1.0f : 0.0f;
		SampleOut<float, 4> out = { boost, d_ori, tryToBite, tryToLay };
		outs.push_back(out);
	}
	NNDataset<float, 11, 4> train_data(ins, outs);

	NNDesc desc = BL_DEFAULT_DESC;

	NeuralNet<float, 11, 4> nn(desc);

	Drawer<float, 11, 4> d(512, nn);

	srand(0);
	while (true) {
		nn.restart = false;
		nn.initialize(0.0f, 1.0f);

		nn.test(train_data);

		std::cout << "training...\n";

		nn.train(train_data, 20000, 1, 0.001f, true);

		if (nn.restart)
			continue;

		nn.test(train_data);

		std::cout << "done.\n";

		nn.save("specimen0.nnn");

		while (!nn.restart)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

int main()
{
	//prepareMinimalSurvivingSpecimen();

	buglife::World w({ 100, 100 });

	w.generate();

	buglife::Drawer d(w, {1024, 1024});

	while (true) {
		w.update();
		std::this_thread::sleep_for(std::chrono::milliseconds(15));
	}
}