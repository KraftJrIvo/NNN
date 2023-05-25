#pragma once

#include <vector>

#include <Eigen/Core>

namespace nnn {

	enum class ActivationFunctionType {
		NONE,
		SIGMOID,
		RELU
	};

	enum class BackPropagationMethod {
		REGULAR,
		FINDIFF,
		FAST
	};

	enum class LossFunctionType {
		L1,
		L2
	};

	struct NNLayerDesc {
		uint64_t sz;
		ActivationFunctionType aft;
	};

	typedef std::vector<NNLayerDesc> NNLayerDescs;
	struct NNDesc {
		NNLayerDescs layers;
		BackPropagationMethod bpt;
		LossFunctionType lft;
	};

	template<typename NN_S, int NN_IN>
	using SampleIn = Eigen::Matrix<NN_S, 1, NN_IN>;
	template<typename NN_S, int NN_OUT>
	using SampleOut = Eigen::Matrix<NN_S, 1, NN_OUT>;
	template<typename NN_S>
	using Sample = Eigen::Matrix<NN_S, 1, -1>;
	template<typename NN_S>
	using Layer = Eigen::Matrix<NN_S, -1, -1>;
	template<typename NN_S, int NN_IN, int NN_OUT>
	struct NNDataset {
		std::vector<SampleIn<NN_S, NN_IN>> ins;
		std::vector<SampleOut<NN_S, NN_OUT>> outs;
		NNDataset(const std::vector<std::vector<NN_S>>& ins_, const std::vector<std::vector<NN_S>>& outs_) {
			auto sz = ins_.size();
			ins.resize(sz);
			outs.resize(sz);
			for (uint64_t i = 0; i < sz; ++i) {
				ins[i] = SampleIn<NN_S, NN_IN>(ins_[i].data());
				outs[i] = SampleOut<NN_S, NN_OUT>(outs_[i].data());
			}
		}
	};

}