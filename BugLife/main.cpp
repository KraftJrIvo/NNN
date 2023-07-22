#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"
#include "World.hpp"
#include "Drawer.h"

using namespace nnn;

// in
// nrg d_nrg closeness r g b target t_ori  (8)
// out
// vel d_ori a_bite a_lay a_target (5)

void prepareMinimalSurvivingSpecimen() {

	std::vector<SampleIn<float, BL_IN_LAYER>> ins;
	std::vector<SampleOut<float, BL_OUT_LAYER>> outs;
	for (int i = 0; i < 10000; ++i) {
		SampleIn<float, BL_IN_LAYER> in;
		for (int j = 0; j < in.cols(); ++j)
			in[j] = BL_RAND_FLOAT;
		float& energy = in[0];
		float& closeness = in[2];
		float& r = in[3]; float& g = in[4]; float& b = in[5];
		bool target = (in[6] > 0.5f);
		float t_ori = in[7];
		
		bool seesFood = target && abs(t_ori - 0.5) < 0.05;
		float boost = seesFood ? 1.0f : 0.75f;
		float d_ori = target ? t_ori : 0.75f;
		float tryToBite = seesFood && (closeness > 0 && closeness <= 0.1);
		float tryToLay = (energy > 0.95) ? 1.0f : 0.0f;
		float tryToTarget = (!target && (r > 0.5 || g > 0.5 || b > 0.5)) ? 1.0f : 0.0f;
		if (tryToBite < 0.5f && i > 9000) {
			i--;
			continue;
		}
		if (tryToLay < 0.5f && i > 8000 && i < 9000) {
			i--;
			continue;
		}
		if (tryToTarget < 0.5f && i > 7000 && i < 8000) {
			i--;
			continue;
		}
		SampleOut<float, BL_OUT_LAYER> out;
		out[0] = boost;
		out[1] = d_ori;
		out[2] = tryToBite;
		out[3] = tryToLay;
		out[4] = tryToTarget;
		ins.push_back(in);
		outs.push_back(out);
	}
	NNDataset<float, BL_IN_LAYER, BL_OUT_LAYER> train_data(ins, outs);

	NNDesc desc = BL_DEFAULT_DESC;

	NeuralNet<float, BL_IN_LAYER, BL_OUT_LAYER> nn(desc);

	Drawer<float, BL_IN_LAYER, BL_OUT_LAYER> d(512, nn);

	srand(0);
	while (true) {
		nn.restart = false;
		nn.initialize(0.0f, 1.0f);

		//nn.test(train_data);

		std::cout << "training...\n";

		nn.train(train_data, 5000, 1, 0.005f, true);

		if (nn.restart)
			continue;

		nn.test(train_data);

		std::cout << "done.\n";

		buglife::Species s; s.brain = nn;
		s.save("zoo/specimen0.spc");

		while (!nn.restart)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

int main()
{
	//prepareMinimalSurvivingSpecimen();

	buglife::World w({ 100, 100 });

	{
		std::ifstream in("world.w", std::ios::binary);
		if (in.good())
			w.load(in);
		else
			w.generate();
	}

	buglife::Drawer d(w, {1024, 1024});

	while (true) {
		w.update();
		//std::this_thread::sleep_for(std::chrono::milliseconds(15));
	}
}