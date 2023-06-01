#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"

using namespace nnn;

#define SCROLL_MARGIN 30

#define IMAGE_SIDE 32
#define IMAGE_VIS_SCALE 4

class SpecificDrawerImage : public SpecificsDrawer<float, 3, 1> {
public:
	SpecificDrawerImage(const std::vector<std::string>& imagePaths) {
		for (auto& p : imagePaths) {
			cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
			cv::resize(img, img, { IMAGE_SIDE * IMAGE_VIS_SCALE, IMAGE_SIDE * IMAGE_VIS_SCALE }, 0.0, 0.0, cv::INTER_NEAREST);
			samples.push_back(img);
		}
	}

	void drawSpecifics(cv::Mat img, NeuralNet<float, 3, 1>& nn, int mx, int my, bool lmb) {
		img.setTo(cv::Scalar(0, 0, 0));
		auto side = img.rows;
		float divC = 1.0f / float(IMAGE_SIDE);

		static float curVal = 0.0f;
		static int counter = 0;
		static bool valChanged = true;

		cv::line(img, { side / 2 + SCROLL_MARGIN, side - SCROLL_MARGIN }, { side - SCROLL_MARGIN, side - SCROLL_MARGIN }, cv::Scalar(125, 125, 125), 2);
		if (lmb && mx > side / 2 + SCROLL_MARGIN && my > side - 2 * SCROLL_MARGIN) {
			float newVal = (mx - (side / 2 + SCROLL_MARGIN)) / float(side / 2 - 2 * SCROLL_MARGIN);
			newVal = std::clamp(newVal, 0.0f, 1.0f);
			valChanged = (newVal != curVal);
			curVal = newVal;
		}
		cv::circle(img, { int((side / 2 + SCROLL_MARGIN) + ((side - SCROLL_MARGIN) - (side / 2 + SCROLL_MARGIN)) * curVal), side - SCROLL_MARGIN }, 4, cv::Scalar(255, 255, 255), -1);
		if (counter % 5 == 0) {
			valChanged = true;
		}

		static SampleIn<float, 3> input;
		static SampleOut<float, 1> output;
		static cv::Mat out_img = cv::Mat(IMAGE_SIDE * IMAGE_VIS_SCALE, IMAGE_SIDE * IMAGE_VIS_SCALE, CV_8UC3);
		if (valChanged) {
			for (int i = 0; i < IMAGE_SIDE; ++i) {
				for (int j = 0; j < IMAGE_SIDE; ++j) {
					input[0] = curVal; input[1] = i * divC; input[2] = j * divC;
					forward(nn, input, output);
					uchar val = uchar(output(0) * 255.0f);
					for (int ii = 0; ii < IMAGE_VIS_SCALE; ++ii) {
						for (int jj = 0; jj < IMAGE_VIS_SCALE; ++jj) {
							out_img.at<cv::Vec3b>(i * IMAGE_VIS_SCALE + ii, j * IMAGE_VIS_SCALE + jj)[0] = val;
							out_img.at<cv::Vec3b>(i * IMAGE_VIS_SCALE + ii, j * IMAGE_VIS_SCALE + jj)[1] = val;
							out_img.at<cv::Vec3b>(i * IMAGE_VIS_SCALE + ii, j * IMAGE_VIS_SCALE + jj)[2] = val;
						}
					}
				}
			}
			valChanged = false;
		}
		cv::Mat in_img = samples[std::min(int(floor(curVal * samples.size())), int(samples.size() - 1))];
		cv::rectangle(img, { img.cols / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE - 1, img.rows / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE / 2 - 1, IMAGE_SIDE * IMAGE_VIS_SCALE + 2, IMAGE_SIDE * IMAGE_VIS_SCALE + 2 }, cv::Scalar(255, 255, 255));
		in_img.copyTo(img({ img.cols / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE, img.rows / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE / 2, IMAGE_SIDE * IMAGE_VIS_SCALE, IMAGE_SIDE * IMAGE_VIS_SCALE }));
		cv::rectangle(img, { img.cols / 2 - 1, img.rows / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE / 2 - 1, IMAGE_SIDE * IMAGE_VIS_SCALE + 2, IMAGE_SIDE * IMAGE_VIS_SCALE + 2 }, cv::Scalar(255, 255, 255));
		out_img.copyTo(img({ img.cols / 2, img.rows / 2 - IMAGE_SIDE * IMAGE_VIS_SCALE / 2, IMAGE_SIDE * IMAGE_VIS_SCALE, IMAGE_SIDE * IMAGE_VIS_SCALE }));
		
		counter++;
	}

private:
	std::vector<cv::Mat> samples;
};

std::shared_ptr<NNDataset<float, 3, 1>> make_dataset(const std::vector<std::string>& imagePaths) {
	std::vector<SampleIn<float, 3>> ins(imagePaths.size() * IMAGE_SIDE * IMAGE_SIDE);
	std::vector<SampleOut<float, 1>> outs(imagePaths.size() * IMAGE_SIDE * IMAGE_SIDE);
	float div = 1.0f / float(std::max(int(imagePaths.size() - 1), 1));
	float divC = 1.0f / float(IMAGE_SIDE);
	int c = 0;
	for (auto& p : imagePaths) {
		cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
		cv::resize(img, img, { IMAGE_SIDE, IMAGE_SIDE });
		for (int i = 0; i < IMAGE_SIDE; ++i) {
			for (int j = 0; j < IMAGE_SIDE; ++j) {
				ins[c * (IMAGE_SIDE * IMAGE_SIDE) + i * IMAGE_SIDE + j] = { c * div, i * divC, j * divC };
				outs[c * (IMAGE_SIDE * IMAGE_SIDE) + i * IMAGE_SIDE + j](0) = float(img.at<uchar>(i, j)) / 255.0f;
			}
		}
		c++;
	}
	return std::make_shared<NNDataset<float, 3, 1>>(ins, outs);
}

void make_video(NeuralNet<float, 3, 1>& nn, size_t nimages, int side, int framesPerImage) {
	cv::VideoWriter outputVideo;
	outputVideo.open("video.mp4", -1, 60, {side, side});
	cv::Mat frame(side, side, CV_8UC1);
	size_t nframes = framesPerImage * nimages;
	float div = 1.0f / float(nframes);
	float divC = 1.0f / float(side);
	float val = 0;
	static SampleIn<float, 3> input;
	static SampleOut<float, 1> output;
	for (int t = 0; t < nframes; ++t) {
		for (int i = 0; i < side; ++i) {
			for (int j = 0; j < side; ++j) {
				input[0] = val; input[1] = i * divC; input[2] = j * divC;
				nn.forward(input, &output);
				uchar val = uchar(output(0) * 255.0f);
				frame.at<uchar>(i, j) = val;
			}
		}
		outputVideo << frame;
		std::cout << t << "/" << nframes << "\r";
		val += div;
	}
}

int main()
{
	std::vector<std::string> imagePaths = { "A.png", "B.png"};
	auto dataset = make_dataset(imagePaths);

	NNDesc desc{
		{
			{3, ActivationFunctionType::NONE},
			{IMAGE_SIDE, ActivationFunctionType::SIGMOID},
			{IMAGE_SIDE / 2, ActivationFunctionType::SIGMOID},
			{1, ActivationFunctionType::SIGMOID}
		},
		OptimizerType::ADAM,
		LossFunctionType::L2
	};

	NeuralNet<float, 3, 1> nn(desc);

	auto sd = std::make_shared<SpecificDrawerImage>(std::vector<std::string>(imagePaths));
	Drawer<float, 3, 1> d(512, nn, sd);

	//srand(0);
	while (true) {
		nn.restart = false;
		nn.initialize(-0.5f, 0.5f);

		nn.test(*dataset);

		std::cout << "training...\n";

		nn.train(*dataset, 5000, 1, 0.001f, true);

		if (nn.restart)
			continue;

		nn.test(*dataset);

		std::cout << "making video...\n";

		sd->paused = true;
		make_video(nn, imagePaths.size(), 256, 60);
		sd->paused = false;

		std::cout << "done.\n";

		while (!nn.restart)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}