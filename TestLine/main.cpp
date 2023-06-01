#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"

using namespace nnn;

#define GRAPH_SCALE 1.0f
#define SCROLL_MARGIN 30
#define OFFSET_BASE 0.05f

float line_coeff = 2.0f;
float line_offset = OFFSET_BASE;

class SpecificDrawerLine : public SpecificsDrawer<float, 1, 1> {
	void drawSpecifics(cv::Mat img, NeuralNet<float, 1, 1>& nn, int mx, int my, bool lmb) {
		img.setTo(cv::Scalar(0, 0, 0));
		auto side = img.rows;
		static float curVal1 = 0.0f;
		static float curVal2 = line_coeff / (GRAPH_SCALE * 2.0f);
		static float curVal3 = (line_offset - OFFSET_BASE) / GRAPH_SCALE;
		float true_w = curVal2 * (GRAPH_SCALE * 2.0f); float true_b = curVal3 * GRAPH_SCALE + OFFSET_BASE;
		float cur_w = getWeights(nn)[0](0, 0); float cur_b = getBiases(nn)[0](0);
		
		float one = float(side) / GRAPH_SCALE;
		cv::line(img, { 0, int(side - one * (true_w * 0 + true_b)) }, { side, int(side - one * (true_w * GRAPH_SCALE + true_b)) }, cv::Scalar(100, 100, 100));
		cv::line(img, { 0, int(side - one * (cur_w * 0 + cur_b)) }, { side, int(side - one * (cur_w * GRAPH_SCALE + cur_b)) }, cv::Scalar(255, 255, 255));

		// scroll 3
		static int counter;
		static bool valChanged = true;
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 2 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			newVal = std::clamp(newVal, 0.0f, 1.0f);
			valChanged = (newVal != curVal1);
			curVal1 = newVal;
		}
		if (counter % 5 == 0) {
			valChanged = true;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal1), side - SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);

		// scroll 2
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - 3 * SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - 3 * SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 4 * SCROLL_MARGIN && my < side - 2 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			curVal2 = std::clamp(newVal, 0.0f, 1.0f);
			line_coeff = curVal2 * GRAPH_SCALE * 2.0f;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal2), side - 3 * SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);

		// scroll 1
		cv::line(img, { side / 2 + SCROLL_MARGIN, side - 5 * SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - 5 * SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 6 * SCROLL_MARGIN && my < side - 4 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			curVal3 = std::clamp(newVal, 0.0f, 1.0f);
			line_offset = curVal3 * GRAPH_SCALE + OFFSET_BASE;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal3), side - 5 * SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);

		static SampleIn<float, 1> input;
		static SampleOut<float, 1> output;
		if (valChanged) {
			input(0) = curVal1 * GRAPH_SCALE;
			forward(nn, input, output);
			valChanged = false;
		}
		cv::circle(img, { int(curVal1 * side), int(side - one * output(0)) }, 4, CV_RGB(255, 0, 0), -1);

		counter++;
	}
};

int main()
{
	NNDesc desc{
		{
			{1, ActivationFunctionType::NONE},
			{1, ActivationFunctionType::NONE}
		},
		OptimizerType::ADAM,
		LossFunctionType::L2
	};

	NeuralNet<float, 1, 1> nn(desc);

	Drawer<float, 1, 1> d(512, nn, std::make_shared<SpecificDrawerLine>());

	srand(0);
	while (true) {
		NNDataset<float, 1, 1> line = {
			{
				{0.0f},
				{0.01f},
				{0.02f},
				{0.03f},
				{0.43f},
				{0.11f},
				{0.12f},
				{0.13f},
			},
			{
				{line_coeff * 0.0f + line_offset},
				{line_coeff * 0.01f + line_offset},
				{line_coeff * 0.02f + line_offset},
				{line_coeff * 0.03f + line_offset},
				{line_coeff * 0.43f + line_offset},
				{line_coeff * 0.11f + line_offset},
				{line_coeff * 0.12f + line_offset},
				{line_coeff * 0.13f + line_offset},
			}
		};

		nn.restart = false;
		nn.initialize(0.0f, 1.0f);

		nn.test(line);

		std::cout << "training...\n";

		nn.train(line, 5000, 8, 0.01f, true, 1);

		if (nn.restart)
			continue;

		nn.test(line);

		std::cout << "done.\n";

		while (!nn.restart)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}