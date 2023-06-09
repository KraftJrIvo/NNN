#pragma once

#include "types.hpp"

namespace nnn::activation {
	
	template<typename Derived>
	void sigmoid(Eigen::MatrixBase<Derived>& x) {
		x = 1.0 / (1.0 + (-x.array()).exp());
	}
	template<typename NN_S>
	NN_S sigmoid_der(NN_S a) {
		return NN_S(1.0) - a;
	}

	template<typename Derived>
	void relu(Eigen::MatrixBase<Derived>& x) {
		x = x.cwiseMax(0.0001f);
	}
	template<typename NN_S>
	NN_S relu_der(NN_S a) {
		return (a > 0) ? NN_S(1.0) : NN_S(0.0001f);
	}

}
