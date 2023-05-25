#pragma once

#include <iomanip>

#include "types.hpp"

#include "act_funcs.hpp"
#include "loss_funcs.hpp"
#include "back_prop.hpp"

namespace nnn {

	template<typename NN_S, int NN_IN, int NN_OUT>
	class NeuralNet
	{
	public:
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
		}

		void initialize(NN_S from, NN_S to) {
			for (uint64_t l = 0; l < _desc.layers.size() - 1; ++l) {
				for (uint64_t j = 0; j < _weights[l].cols(); ++j) {
					for (uint64_t i = 0; i < _weights[l].rows(); ++i) {
						_weights[l](i,j) = from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
					}
					_biases[l](j) = from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
				}
			}
		}

		void backPropAndLearn(const NNLayerDescs& layers, const SampleOut<NN_S, NN_OUT>& out, uint64_t batch_sz, NN_S learn_rate) {
			auto sz = _desc.layers.size();
			if (!_d_weights.size()) {
				_d_weights.resize(sz - 1);
				_d_biases.resize(sz - 1);
				_d_activations.resize(sz);
				_d_activations[0].resize(1, _desc.layers[0].sz);
				for (uint64_t i = 1; i < sz; ++i) {
					_d_weights[i - 1].resize(_desc.layers[i - 1].sz, _desc.layers[i].sz);
					_d_biases[i - 1].resize(1, _desc.layers[i].sz);
					_d_activations[i].resize(1, _desc.layers[i].sz);
				}
			}
			_d_activations[0].setZero();
			for (uint64_t i = 1; i < sz; ++i) {
				_d_weights[i - 1].setZero();
				_d_biases[i - 1].setZero();
				_d_activations[i].setZero();
			}

			switch (_desc.bpt)
			{
			case BackPropagationMethod::REGULAR:
				back_prop::regular<NN_S, NN_OUT>(layers, out, batch_sz, learn_rate, _weights, _biases, _activations, _d_weights, _d_biases, _d_activations);
				break;
			default:
				break;
			}
		}

		void train(const NNDataset<NN_S, NN_IN, NN_OUT>& dataset, uint64_t n_epochs, uint64_t batch_sz, NN_S learn_rate, bool stohastic = false) {
			uint64_t sz = dataset.ins.size();
			uint64_t n_batches = ceil(sz / batch_sz);
			std::vector<SampleOut<NN_S, NN_OUT>> outBatch(batch_sz);
			for (uint64_t e = 0; e < n_epochs; ++e) {
				NN_S loss = 0;
				for (uint64_t b = 0; b < n_batches; ++b) {
					for (uint64_t s = 0; s < batch_sz; ++s) {
						uint64_t r = batch_sz * b + s;
						auto sample = dataset.ins[r];
						outBatch[s] = dataset.outs[r];
						forward(sample);
						loss += computeLoss(dataset.outs[r]);
						backPropAndLearn(_desc.layers, dataset.outs[r], batch_sz, learn_rate);
					}
				}
				loss /= NN_S(batch_sz * n_batches);
				std::cout << "epoch " << e << "/" << n_epochs << " loss: " << loss << "\n";
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
				break;
			}
		}

		NN_S loss(const Sample<NN_S>& input) {
			switch (_desc.lft)
			{
			case LossFunctionType::SQUARE:
				return loss::square(input);
			default:
				break;
			}
			return INFINITY;
		}

		NN_S computeLoss(const SampleOut<NN_S, NN_OUT>& output) {
			return loss(getResult() - output);
		}

		void forward(const SampleIn<NN_S, NN_IN>& input) {
			_activations[0] = input;
			for (uint64_t i = 0; i < _weights.size(); ++i) {
				_activations[i + 1] = _activations[i] * _weights[i] + _biases[i];
				activation(_desc.layers[i + 1].aft, _activations[i + 1]);
			}			
		}

		void test(const NNDataset<NN_S, NN_IN, NN_OUT>& dataset) {
			std::cout << std::fixed;
			std::cout << std::setprecision(3);
			NN_S loss = 0;
			for (uint64_t s = 0; s < dataset.ins.size(); ++s) {
				std::cout << "[" << dataset.ins[s] << "]: [";
				forward(dataset.ins[s]);
				loss += computeLoss(dataset.outs[s]);
				std::cout << getResult() << "] / [" << dataset.outs[s] << "]\n";
			}
			std::cout << "loss: " << (loss / NN_S(dataset.ins.size())) << "\n";
		}

		SampleOut<NN_S, NN_OUT> getResult() {
			return _activations[_activations.size() - 1];
		}
		 
	private:
		NNDesc _desc;
		std::vector<Layer<NN_S>> _weights;
		std::vector<Sample<NN_S>> _biases;
		std::vector<Sample<NN_S>> _activations;
		std::vector<Layer<NN_S>> _d_weights;
		std::vector<Sample<NN_S>> _d_biases;
		std::vector<Sample<NN_S>> _d_activations;
	};

}