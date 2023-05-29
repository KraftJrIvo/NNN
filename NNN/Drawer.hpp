#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "NeuralNet.hpp"

#define MARGIN 0.05f

namespace nnn {

	template<typename NN_S, int NN_IN, int NN_OUT>
	class SpecificsDrawer {
	public:
		virtual void drawSpecifics(cv::Mat img, NeuralNet<NN_S, NN_IN, NN_OUT>& nn, int mx, int my, bool lmb) = 0;
		std::vector<Layer<NN_S>>& getWeights(NeuralNet<NN_S, NN_IN, NN_OUT>& nn) { return nn._weights; }
		std::vector<Sample<NN_S>>& getBiases(NeuralNet<NN_S, NN_IN, NN_OUT>& nn) { return nn._biases; }
		void forward(NeuralNet<NN_S, NN_IN, NN_OUT>& nn, const SampleIn<NN_S, NN_IN>& input, SampleOut<NN_S, NN_OUT>& out) { nn.lock(); nn.forward(input, &out); nn.unlock(); }
		const SampleIn<NN_S, NN_IN>& getOutput(NeuralNet<NN_S, NN_IN, NN_OUT>& nn) { return nn._activations[nn._activations.size() - 1]; }
	};

	template<typename NN_S, int NN_IN, int NN_OUT>
	class Drawer {		
	public:
		Drawer(uint32_t side, NeuralNet<NN_S, NN_IN, NN_OUT>& nn, std::shared_ptr<SpecificsDrawer<NN_S, NN_IN, NN_OUT>> sd = nullptr) :
		_side(side),
		_nn(nn),
		_sd(sd)
		{
			_img.create(side, (_sd ? 3 : 2) * side, CV_8UC3);

			_redrawer = std::thread([&]() {
				while (true) {
					_draw();
					std::this_thread::sleep_for(std::chrono::milliseconds(15));
				}
			});
		}

		void reset() {
			_lossVals.clear();
			_maxLoss = 0;
		}
	private:
		void _drawLoss(cv::Mat img, float loss) {

			if (loss > _maxLoss) {
				_maxLoss = loss;
			}
			int pixLoss = int((loss / _maxLoss) * _side * (1.0f - 2.0f * MARGIN));
			_lossVals.push_back(pixLoss);

			img.setTo(cv::Scalar(0, 0, 0));
			cv::line(img, { int(_side * MARGIN - 1), int(_side * MARGIN) }, { int(_side * MARGIN - 1), int(_side * (1.0f - MARGIN) + 1) }, cv::Scalar(255, 255, 255));
			cv::line(img, { int(_side * MARGIN - 1), int(_side * (1.0f - MARGIN) + 1) }, { int(_side * (1.0f - MARGIN) + 1), int(_side * (1.0f - MARGIN) + 1) }, cv::Scalar(255, 255, 255));
			cv::Point prev;
			for (uint64_t i = 0; i < _lossVals.size(); ++i) {
				auto t = (1.0f - 2.0f * MARGIN) / float(_lossVals.size());
				auto t2 = float(_side) * MARGIN + float(i) * t;
				cv::Point p = { int(float(_side) * (MARGIN + float(i) * (1.0f - 2.0f * MARGIN) / float(_lossVals.size()))), int(_side * (1.0f - MARGIN) - _lossVals[i])};
				cv::line(img, p, (i == 0) ? p : prev, CV_RGB(255, 0, 0));
				prev = p;
			}
		}

		void _drawNeurons(cv::Mat img) {
			img.setTo(cv::Scalar(30, 30, 30));
			int s = int(_side * 1.0f - 2.0f * MARGIN);
			int divH = int(s / _nn._desc.layers.size());
			int prevDivV = 0;
			for (uint64_t i = 0; i < _nn._desc.layers.size(); ++i) {
				auto& l = _nn._desc.layers[i];
				int divV = int(s / l.sz);
				if (i == 0) {
					for (uint64_t j = 0; j < _nn._desc.layers[i].sz; ++j) {
						int b = int(255.0f * float(_nn._activations[0](j)));
						cv::circle(img, { int(_side * MARGIN), int(_side * MARGIN + divV * (j + 0.5)) }, 3, cv::Scalar(b, b, b), -1);
					}
				} else {
					for (uint64_t j = 0; j < _nn._desc.layers[i].sz; ++j) {
						for (uint64_t k = 0; k < _nn._desc.layers[i - 1].sz; ++k) {
							int b = int(255.0f * float(_nn._weights[i - 1](k, j)));
							cv::line(img, { int(_side * MARGIN + divH * (i - 1)), int(_side * MARGIN + prevDivV * (k + 0.5)) }, { int(_side * MARGIN + divH * i), int(_side * MARGIN + divV * (j + 0.5)) }, cv::Scalar(b, b, b));
						}
						int b = int(255.0f * float(_nn._activations[i](j))); 
						if (i == _nn._desc.layers.size() - 1) {
							cv::line(img, { int(_side * MARGIN + divH * i), int(_side * MARGIN + divV * (j + 0.5)) }, { int(_side * MARGIN + divH * (i + 1)), int(_side * MARGIN + divV * (j + 0.5)) }, cv::Scalar(b, b, b));
						}
						b = int(255.0f * float(_nn._biases[i - 1](j)));
						cv::circle(img, { int(_side * MARGIN + divH * i), int(_side * MARGIN + divV * (j + 0.5)) }, 6, cv::Scalar(b, b, b), -1);
					}
				}
				prevDivV = divV;
			}
		}

		void _draw() {

			auto e = _nn.getCurrentEpoch();
			if (e != _lastEpoch) {
				_drawLoss(_img({ 0, 0, _side, _side }), _nn.getCurrentLoss());
				_lastEpoch = e;
			}
			_drawNeurons(_img({ _side, 0, _side, _side}));
			if (_sd) {
				_sd->drawSpecifics(_img({ 2 * _side, 0, _side, _side }), _nn, _mouse.x - 2 * _side, _mouse.y, lmb);
			}

			auto onMouse = [](int event, int x, int y, int flags, void* _data) {
				static bool lmb = false;

				((Drawer*)_data)->_mouse.x = x;
				((Drawer*)_data)->_mouse.y = y;

				static cv::Point2f lastPos = { (float)x, (float)y };

				if (event == cv::EVENT_LBUTTONDOWN && !lmb) {
					lastPos = { (float)x, (float)y };
				}

				if (event == cv::EVENT_LBUTTONUP && lmb) {
					
				}

				if (event == cv::EVENT_LBUTTONDOWN) lmb = true;
				if (event == cv::EVENT_LBUTTONUP) lmb = false;

				((Drawer*)_data)->lmb = lmb;

				static bool rmb = false;
				if (event == cv::EVENT_RBUTTONDOWN) rmb = true;
				if (event == cv::EVENT_RBUTTONUP) rmb = false;
			};

			cv::imshow("NNN", _img);
			auto k = cv::waitKey(1);
			bool space = (k % 256 == 32);
			if (space) {
				_nn.restart = true;
				reset();
			}
			cv::setMouseCallback("NNN", onMouse, this);
		}

	private:
		NeuralNet<NN_S, NN_IN, NN_OUT>& _nn;
		int _side;
		cv::Mat _img;
		std::thread _redrawer;
		cv::Point2i _mouse;
		bool lmb = false;

		uint64_t _lastEpoch = 0;
		std::vector<int> _lossVals;
		float _maxLoss = 0;

		std::shared_ptr<SpecificsDrawer<NN_S, NN_IN, NN_OUT>> _sd;
	};

}
