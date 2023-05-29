#pragma once

#include <iomanip>
#include <random>
#include <algorithm>

#include "types.hpp"

#include "act_funcs.hpp"
#include "loss_funcs.hpp"
#include "back_prop.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace nnn {
	template<typename NN_S, int NN_IN, int NN_OUT>
	class Drawer;
	template<typename NN_S, int NN_IN, int NN_OUT>
	class SpecificsDrawer;

	template<typename NN_S, int NN_IN, int NN_OUT>
	class NeuralNet
	{
	template<typename, int, int> friend class Drawer;
	template<typename, int, int> friend class SpecificsDrawer;
	public:
		bool restart = false;
		uint64_t inSz = 0, outSz = 0, maxSz = 0;

		NeuralNet(const NNDesc& desc) :
			_desc(desc)
		{
			auto sz = _desc.layers.size();
			_weights.resize(sz - 1);
			_biases.resize(sz - 1);
			_activations.resize(sz);
			_activations[0].resize(1, _desc.layers[0].sz);
			for (uint64_t i = 1; i < sz; ++i) {
				_weights[i - 1].resize(_desc.layers[i - 1].sz, _desc.layers[i].sz);
				_biases[i - 1].resize(1, _desc.layers[i].sz);
				_activations[i].resize(1, _desc.layers[i].sz);
				maxSz = std::max(maxSz, _desc.layers[i].sz);
			}
			_d_weights.resize(sz - 1);
			_d_biases.resize(sz - 1);
			_d_activations.resize(sz);
			_d_activations[0].resize(1, _desc.layers[0].sz);
			for (uint64_t i = 1; i < sz; ++i) {
				_d_weights[i - 1].resize(_desc.layers[i - 1].sz, _desc.layers[i].sz);
				_d_biases[i - 1].resize(1, _desc.layers[i].sz);
				_d_activations[i].resize(1, _desc.layers[i].sz);
			}
			resetGrad();
		}

		void initialize(NN_S from, NN_S to) {
			for (uint64_t l = 0; l < _desc.layers.size() - 1; ++l) {
				for (int64_t j = 0; j < _weights[l].cols(); ++j) {
					for (int64_t i = 0; i < _weights[l].rows(); ++i) {
						_weights[l](i,j) = from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
					}
					_biases[l](j) = from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
				}
			}
		}

		void backProp(const NNLayerDescs& layers, const SampleOut<NN_S, NN_OUT>& out, uint64_t batch_sz, float curLossCoeff) {
			switch (_desc.bpt)
			{
			case BackPropagationMethod::REGULAR:
				back_prop::regular<NN_S, NN_OUT>(layers, out, batch_sz, _weights, _biases, _activations, _d_weights, _d_biases, _d_activations);
				break;
			case BackPropagationMethod::ADAPTIVE:
				back_prop::adaptive<NN_S, NN_OUT>(layers, out, batch_sz, curLossCoeff, _weights, _biases, _activations, _d_weights, _d_biases, _d_activations);
			default:
				break;
			}
		}

		void resetGrad() {
			auto sz = _desc.layers.size();
			_d_activations[0].setZero();
			for (uint64_t i = 1; i < sz; ++i) {
				_d_weights[i - 1].setZero();
				_d_biases[i - 1].setZero();
				_d_activations[i].setZero();
			}
		}

		void learn(NN_S learn_rate) {
			auto sz = _desc.layers.size();
			for (uint64_t l = 0; l < sz - 1; ++l) {
				_weights[l] -= learn_rate * _d_weights[l];
				_biases[l] -= learn_rate * _d_biases[l];
			}
			resetGrad();
		}

		void train(const NNDataset<NN_S, NN_IN, NN_OUT>& dataset, uint64_t n_epochs, uint64_t batch_sz, NN_S learn_rate, bool stochastic = false, uint32_t holdback_ms = 0) {

			uint64_t sz = dataset.ins.size();

			std::vector<int> indices(sz);
			for (int i = 0; i < sz; ++i)
				indices[i] = i;

			uint64_t n_batches = uint64_t(ceil(sz / batch_sz));
			for (uint64_t e = 0; e < n_epochs; ++e) {
				NN_S loss = 0, curLoss;
				if (stochastic) {
					std::random_device rd;
					std::mt19937 g(rd());
					std::shuffle(indices.begin(), indices.end(), g);
				}
				NN_S maxLoss = -INFINITY;
				for (uint64_t b = 0; b < n_batches; ++b) {
					for (uint64_t s = 0; s < batch_sz; ++s) {
						uint64_t r = indices[(batch_sz * b + s) % sz];
						auto sample = dataset.ins[r];
						if (restart) break;
						lock();
						forward(sample);
						curLoss = computeLoss(dataset.outs[r]);
						maxLoss = std::max(maxLoss, curLoss);
						loss += curLoss;
						backProp(_desc.layers, dataset.outs[r], batch_sz, curLoss / maxLoss);
						unlock();
					}
					if (restart) break;
					learn(learn_rate);
				}
				loss /= NN_S(batch_sz * n_batches);
				if (e % 10 == 0) {
					std::cout << "epoch " << e << "/" << n_epochs << " loss: " << loss << "\n";
				}
				_lastLoss = loss;
				_lastEpoch = e;
				if (holdback_ms) {
					std::this_thread::sleep_for(std::chrono::milliseconds(holdback_ms));
				}
				if (restart) break;
			}
		}

		void activation(ActivationFunctionType aft, Sample<NN_S>& input) {
			switch (aft)
			{
			case ActivationFunctionType::SIGMOID:
				activation::sigmoid(input);
				break;
			case ActivationFunctionType::RELU:
				activation::relu(input);
				break;
			default:
				input = input.array().abs();
				break;
			}
		}

		NN_S loss(const Sample<NN_S>& input) {
			switch (_desc.lft)
			{
			case LossFunctionType::L1:
				return loss::l1(input);
			case LossFunctionType::L2:
				return loss::l2(input);
			default:
				break;
			}
			return INFINITY;
		}

		NN_S computeLoss(const SampleOut<NN_S, NN_OUT>& output) {
			return loss(getOutput() - output);
		}

		void forward(const SampleIn<NN_S, NN_IN>& input = {}, SampleOut<NN_S, NN_OUT>* output = nullptr) {
			if (input.cols())
				_activations[0] = input;
			for (uint64_t i = 0; i < _weights.size(); ++i) {
				_activations[i + 1] = _activations[i] * _weights[i] + _biases[i];
				activation(_desc.layers[i + 1].aft, _activations[i + 1]);
			}
			if (output)
				*output = getOutput();
		}

		void test(const NNDataset<NN_S, NN_IN, NN_OUT>& dataset) {
			std::cout << std::fixed;
			std::cout << std::setprecision(3);
			NN_S loss = 0;
			SampleOut<NN_S, NN_OUT> out;
			for (uint64_t s = 0; s < dataset.ins.size(); ++s) {
				std::cout << "[" << dataset.ins[s] << "]: [";
				forward(dataset.ins[s], &out);
				loss += computeLoss(dataset.outs[s]);
				std::cout << out << "] / [" << dataset.outs[s] << "]\n";
			}
			std::cout << "loss: " << (loss / NN_S(dataset.ins.size())) << "\n";
		}

		SampleOut<NN_S, NN_OUT> getOutput() {
			return _activations[_activations.size() - 1];
		}

		float getCurrentLoss() {
			return _lastLoss;
		}

		uint64_t getCurrentEpoch() {
			return _lastEpoch;
		}

		void lock() {
			_forward_lock.lock();
		}

		void unlock() {
			_forward_lock.unlock();
		}

	private:
		NNDesc _desc;
		std::vector<Layer<NN_S>> _weights;
		std::vector<Sample<NN_S>> _biases;
		std::vector<Sample<NN_S>> _activations;
		std::vector<Layer<NN_S>> _d_weights;
		std::vector<Sample<NN_S>> _d_biases;
		std::vector<Sample<NN_S>> _d_activations;

		std::mutex _forward_lock;

		float _lastLoss = 0.0f;
		uint64_t _lastEpoch = 0;
	};

}