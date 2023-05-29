#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"

using namespace nnn;

#define SCROLL_MARGIN 30

class SpecificDrawerLogic : public SpecificsDrawer<float, 3, 1> {
	void drawSpecifics(cv::Mat img, NeuralNet<float, 3, 1>& nn, int mx, int my, bool lmb) {
		img.setTo(cv::Scalar(0, 0, 0));
		auto side = img.rows;
		static float curVal1 = 0.0f;
		static float curVal2 = 0.0f;
		static float curVal3 = 0.0f;
		static SampleIn<float, 3> input;
		static SampleOut<float, 1> output;
		static int counter;
		static bool valChanged = true;
		if (valChanged) {
			input = { curVal1, curVal2, curVal3 };
			forward(nn, input, output);
			valChanged = false;
		}

		cv::circle(img, { side / 3, side / 3 }, side / 20, cv::Scalar(255 * curVal2, 255 * curVal2, 255 * curVal2), -1);
		cv::circle(img, { side / 3, side / 3 }, side / 20, cv::Scalar(255, 255, 255));
		cv::line(img, { side / 3, side / 3 }, { 2 * side / 3, side / 2 }, cv::Scalar(255 * curVal2, 255 * curVal2, 255 * curVal2));
		cv::circle(img, { side / 3, 2 * side / 3 }, side / 20, cv::Scalar(255 * curVal3, 255 * curVal3, 255 * curVal3), -1);
		cv::circle(img, { side / 3, 2 * side / 3 }, side / 20, cv::Scalar(255, 255, 255));
		cv::line(img, { side / 3, 2 * side / 3 }, { 2 * side / 3, side / 2 }, cv::Scalar(255 * curVal3, 255 * curVal3, 255 * curVal3));
		cv::circle(img, { 2 * side / 3, side / 2 }, side / 10, cv::Scalar(255 * output(0), 255 * output(0), 255 * output(0)), -1);
		cv::circle(img, { 2 * side / 3, side / 2 }, side / 10, cv::Scalar(255, 255, 255));

		// scroll 3
		static std::vector<std::string> txt = {"not", "and", "xor", "or", "yes"};
		static int div = (side / 2 - 2 * SCROLL_MARGIN) / 4;
		for (int i = 0; i < 5; ++i) {
			if (i < 4) 
				cv::line(img, { side / 2 + SCROLL_MARGIN + i * div + div / 2, side - 5 * SCROLL_MARGIN - 5 }, { side / 2 + SCROLL_MARGIN + i * div + div / 2, side - 5 * SCROLL_MARGIN + 5 }, cv::Scalar(125, 125, 125));
			cv::putText(img, txt[i], { int(side / 2 + SCROLL_MARGIN + i * div * 0.9f), side - 5 * SCROLL_MARGIN - 5 }, cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(125, 125, 125));
		}
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 2 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			newVal = std::clamp(newVal, 0.0f, 1.0f);
			valChanged = (newVal != curVal3);
			curVal3 = newVal;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal3), side - SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);
		// scroll 2
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - 3 * SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - 3 * SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 4 * SCROLL_MARGIN && my < side - 2 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			curVal2 = std::clamp(newVal, 0.0f, 1.0f);
			newVal = std::clamp(newVal, 0.0f, 1.0f);
			valChanged = (newVal != curVal2);
			curVal2 = newVal;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal2), side - 3 * SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);
		// scroll 1
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - 5 * SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - 5 * SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 6 * SCROLL_MARGIN && my < side - 4 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			newVal = std::clamp(newVal, 0.0f, 1.0f);
			valChanged = (newVal != curVal1);
			curVal1 = newVal;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal1), side - 5 * SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);
		if (counter % 5 == 0) {
			valChanged = true;
		}

		counter++;
	}
};

int main()
{
	NNDataset<float, 3, 1> log_ops = {
		{
			{0.0f, 0.0f, 0.0f},
			{0.0f, 0.0f, 1.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 1.0f, 1.0f},
			{0.25f, 0.0f, 0.0f},
			{0.25f, 0.0f, 1.0f},
			{0.25f, 1.0f, 0.0f},
			{0.25f, 1.0f, 1.0f},
			{0.5f, 0.0f, 0.0f},
			{0.5f, 0.0f, 1.0f},
			{0.5f, 1.0f, 0.0f},
			{0.5f, 1.0f, 1.0f},
			{0.75f, 0.0f, 0.0f},
			{0.75f, 0.0f, 1.0f},
			{0.75f, 1.0f, 0.0f},
			{0.75f, 1.0f, 1.0f},
			{1.0f, 0.0f, 0.0f},
			{1.0f, 0.0f, 1.0f},
			{1.0f, 1.0f, 0.0f},
			{1.0f, 1.0f, 1.0f},
			{0.5f, 1.0f, 1.0f},
			{0.25f, 1.0f, 1.0f},
			{0.5f, 1.0f, 1.0f},
			{0.25f, 1.0f, 1.0f},
			{0.5f, 1.0f, 1.0f},
			{0.25f, 1.0f, 1.0f},
			{0.5f, 1.0f, 1.0f},
			{0.25f, 1.0f, 1.0f},
		},
		{
			{0.0f},
			{0.0f},
			{0.0f},
			{0.0f},
			{0.0f},
			{0.0f},
			{0.0f},
			{1.0f},
			{0.0f},
			{1.0f},
			{1.0f},
			{0.0f},
			{0.0f},
			{1.0f},
			{1.0f},
			{1.0f},
			{1.0f},
			{1.0f},
			{1.0f},
			{1.0f},
			{0.0f},
			{1.0f},
			{0.0f},
			{1.0f},
			{0.0f},
			{1.0f},
			{0.0f},
			{1.0f},
		}
	};

	NNDesc desc{
		{
			{3, ActivationFunctionType::NONE}, 
			{3, ActivationFunctionType::SIGMOID},
			{2, ActivationFunctionType::SIGMOID},
			{1, ActivationFunctionType::SIGMOID}
		}, 
		BackPropagationMethod::ADAPTIVE,
		LossFunctionType::L2
	};

	NeuralNet<float, 3, 1> nn(desc);

	Drawer<float, 3, 1> d(512, nn, std::make_shared<SpecificDrawerLogic>());

	while (true) {
		//srand(0);
		nn.restart = false;
		nn.initialize(0.0f, 1.0f);

		nn.test(log_ops);

		std::cout << "training...\n";

		nn.train(log_ops, 100000, 1, 0.01f, true);

		if (nn.restart)
			continue;

		nn.test(log_ops);

		std::cout << "done.\n";

		while (!nn.restart)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}