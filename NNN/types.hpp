#pragma once

#include <vector>
#include <fstream>

#include <Eigen/Core>

namespace nnn {

	enum class ActivationFunctionType {
		NONE,
		SIGMOID,
		RELU
	};

	enum class OptimizerType {
		SIMPLE,
		ADAM
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
		OptimizerType opt;
		LossFunctionType lft;
		void save(std::ofstream& out) {
			size_t sz = layers.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& l : layers) {
				out.write(reinterpret_cast<const char*>(&l.sz), sizeof(uint64_t));
				out.write(reinterpret_cast<const char*>(&l.aft), sizeof(ActivationFunctionType));
			}
			out.write(reinterpret_cast<const char*>(&opt), sizeof(OptimizerType));
			out.write(reinterpret_cast<const char*>(&lft), sizeof(LossFunctionType));
		}
		void load(std::ifstream& in) {
			size_t sz; in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); layers.resize(sz);
			for (auto& l : layers) {
				in.read(reinterpret_cast<char*>(&l.sz), sizeof(uint64_t));
				in.read(reinterpret_cast<char*>(&l.aft), sizeof(ActivationFunctionType));
			}
			in.read(reinterpret_cast<char*>(&opt), sizeof(OptimizerType));
			in.read(reinterpret_cast<char*>(&lft), sizeof(LossFunctionType));
		}
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
		NNDataset(const std::vector<SampleIn<NN_S, NN_IN>>& ins_, const std::vector<SampleOut<NN_S, NN_OUT>>& outs_) :
			ins(ins_),
			outs(outs_)
		{
			auto sz = ins.size();
		}
	};

}