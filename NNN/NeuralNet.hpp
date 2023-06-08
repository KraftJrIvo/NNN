#pragma once

#include <iomanip>
#include <random>
#include <algorithm>

#include "types.hpp"

#include "act_funcs.hpp"
#include "loss_funcs.hpp"

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

		NeuralNet(const NeuralNet<NN_S, NN_IN, NN_OUT>& nn) {
			create(nn._desc);
			_weights = nn._weights;
			_biases = nn._biases;
		}

		NeuralNet(const NNDesc& desc) {
			create(desc);
		}

		NeuralNet(const std::string& filePath) {
			load(filePath);
		}

		void load(const std::string& filePath) {
			std::ifstream in(filePath, std::ios::binary);
			NNDesc desc; desc.load(in);
			create(desc);
			for (int64_t l = 0; l < _desc.layers.size() - 1; ++l) {
				for (int64_t j = 0; j < _biases[l].cols(); ++j) {
					in.read(reinterpret_cast<char*>(&_biases[l](j)), sizeof(NN_S));
					for (int64_t i = 0; i < _weights[l].rows(); ++i) {
						in.read(reinterpret_cast<char*>(&_weights[l](i, j)), sizeof(NN_S));
					}
				}
			}
		}

		void save(const std::string& filePath) {
			std::ofstream out(filePath, std::ios::binary);
			_desc.save(out);
			for (int64_t l = 0; l < _desc.layers.size() - 1; ++l) {
				for (int64_t j = 0; j < _biases[l].cols(); ++j) {
					out.write(reinterpret_cast<const char*>(&_biases[l](j)), sizeof(NN_S));
					for (int64_t i = 0; i < _weights[l].rows(); ++i) {
						out.write(reinterpret_cast<const char*>(&_weights[l](i, j)), sizeof(NN_S));
					}
				}
			}
		}

		void create(const NNDesc& desc) {
			_desc = desc;
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
			if (_desc.opt == OptimizerType::ADAM) {
				_adam_m_weights.resize(sz - 1);
				_adam_m_biases.resize(sz - 1);
				_adam_v_weights.resize(sz - 1);
				_adam_v_biases.resize(sz - 1);
				for (uint64_t i = 1; i < sz; ++i) {
					_adam_m_weights[i - 1].resize(_desc.layers[i - 1].sz, _desc.layers[i].sz);
					_adam_m_biases[i - 1].resize(1, _desc.layers[i].sz);
					_adam_v_weights[i - 1].resize(_desc.layers[i - 1].sz, _desc.layers[i].sz);
					_adam_v_biases[i - 1].resize(1, _desc.layers[i].sz);
				}
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

		void backProp(const SampleOut<NN_S, NN_OUT>& out, uint64_t batch_sz) {

			auto n_layers = _activations.size();
			_d_activations[n_layers - 1] = NN_S(2) / NN_S(batch_sz) * (_activations[n_layers - 1] - out);

			for (uint64_t l = n_layers - 1; l > 0; --l) {
				for (int64_t i = 0; i < _activations[l].cols(); ++i) {
					NN_S a = _activations[l](i);
					NN_S da = _d_activations[l](i);
					_d_biases[l - 1](i) += da * a * activation_der(_desc.layers[l].aft, a);
					for (int64_t j = 0; j < _activations[l - 1].cols(); ++j) {
						NN_S pa = _activations[l - 1](j);
						NN_S w = _weights[l - 1](j, i);
						_d_weights[l - 1](j, i) += da * a * activation_der(_desc.layers[l].aft, a) * pa;
						_d_activations[l - 1](j) += da * a * activation_der(_desc.layers[l].aft, a) * w;
					}
				}
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

		void resetAdam() {
			auto sz = _desc.layers.size();
			for (uint64_t i = 0; i < sz - 1; ++i) {
				_adam_m_weights[i].setZero();
				_adam_v_weights[i].setZero();
				_adam_m_biases[i].setZero();
				_adam_v_biases[i].setZero();
			}
		}

		void learn(NN_S learn_rate) {
			auto sz = _desc.layers.size();
			for (uint64_t l = 0; l < sz - 1; ++l) {
				_weights[l] -= learn_rate * _d_weights[l];
				_biases[l] -= learn_rate * _d_biases[l];
			}
		}

		void learnAdam(NN_S learn_rate) {
			
			NN_S b1 = 0.9f, b2 = 0.999f, eps = 1e-08f;

			int64_t sz = _desc.layers.size();
			for (int64_t l = 0; l < sz - 1; ++l) {
				for (int64_t j = 0; j < _weights[l].cols(); ++j) {
					_adam_m_biases[l](j) = b1 * _adam_m_biases[l](j) + (1.0f - b1) * _d_biases[l](j);
					_adam_v_biases[l](j) = b2 * _adam_v_biases[l](j) + (1.0f - b2) * _d_biases[l](j) * _d_biases[l](j);
					for (int64_t i = 0; i < _weights[l].rows(); ++i) {
						_adam_m_weights[l](i, j) = b1 * _adam_m_weights[l](i, j) + (1.0f - b1) * _d_weights[l](i, j);
						_adam_v_weights[l](i, j) = b2 * _adam_v_weights[l](i, j) + (1.0f - b2) * _d_weights[l](i, j) * _d_weights[l](i, j);
					}
					_biases[l](j) -= learn_rate * (_adam_m_biases[l](j) / (1.0f - b1)) / (sqrt(_adam_v_biases[l](j) / (1.0f - b2)) + eps);
					for (int64_t i = 0; i < _weights[l].rows(); ++i) {
						_weights[l](i, j) -= learn_rate * (_adam_m_weights[l](i, j) / (1.0f - b1)) / (sqrt(_adam_v_weights[l](i, j) / (1.0f - b2)) + eps);
					}
				}
			}
		}

		void train(const NNDataset<NN_S, NN_IN, NN_OUT>& dataset, uint64_t n_epochs, uint64_t batch_sz, NN_S learn_rate, bool stochastic = false, uint32_t holdback_ms = 0) {
			
			if (_desc.opt == OptimizerType::ADAM) {
				resetAdam();
			}

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
				NN_S maxLoss = NN_S(1e-10);
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
						backProp(dataset.outs[r], batch_sz);
						unlock();
					}
					if (restart) break;
					if (_desc.opt == OptimizerType::ADAM)
						learnAdam(learn_rate);
					else
						learn(learn_rate);
					resetGrad();
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

		NN_S activation_der(ActivationFunctionType aft, NN_S a) {
			switch (aft)
			{
			case ActivationFunctionType::SIGMOID:
				return activation::sigmoid_der(a);
			case ActivationFunctionType::RELU:
				return activation::relu_der(a);
			default:
				return 1;
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

		std::vector<Layer<NN_S>> _adam_m_weights;
		std::vector<Sample<NN_S>> _adam_m_biases;
		std::vector<Layer<NN_S>> _adam_v_weights;
		std::vector<Sample<NN_S>> _adam_v_biases;

		std::mutex _forward_lock;

		float _lastLoss = 0.0f;
		uint64_t _lastEpoch = 0;
	};

}