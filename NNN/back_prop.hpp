#pragma once

#include "types.hpp"

#include "act_funcs.hpp"

namespace nnn::back_prop {

	template<typename NN_S>
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

	template<typename NN_S, int NN_OUT>
	void regular(const NNLayerDescs& layers, const SampleOut<NN_S, NN_OUT>& out, uint64_t batch_sz, NN_S learn_rate,
		std::vector<Layer<NN_S>>& weights, std::vector<Sample<NN_S>>& biases, std::vector<Sample<NN_S>>& activations,
		std::vector<Layer<NN_S>>& d_weights, std::vector<Sample<NN_S>>& d_biases, std::vector<Sample<NN_S>>& d_activations) {

		auto n_layers = activations.size();

		d_activations[n_layers - 1] = NN_S(2) / NN_S(batch_sz) * (activations[n_layers - 1] - out);

		for (uint64_t l = n_layers - 1; l > 0; --l) {
			for (int64_t i = 0; i < activations[l].cols(); ++i) {
				NN_S a = activations[l](i);
				NN_S da = d_activations[l](i);
				d_biases[l - 1](i) += da * a * activation_der(layers[l].aft, a);
				for (int64_t j = 0; j < activations[l - 1].cols(); ++j) {
					NN_S pa = activations[l - 1](j);
					NN_S w = weights[l - 1](j, i);
					d_weights[l - 1](j, i) += da * a * activation_der(layers[l].aft, a) * pa;
					d_activations[l - 1](j) += da * a * activation_der(layers[l].aft, a) * w;
				}
			}
		}

		for (uint64_t l = 0; l < n_layers - 1; ++l) {
			weights[l] -= learn_rate * d_weights[l];
			biases[l] -= learn_rate * d_biases[l];
		}
	}

}
