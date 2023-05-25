#pragma once

#include "types.hpp"

namespace nnn::loss {

	template<typename Derived>
	typename Eigen::MatrixBase<Derived>::Scalar l1(const Eigen::MatrixBase<Derived>& x) {
		return sqrt(x.dot(x));
	}

	template<typename Derived>
	typename Eigen::MatrixBase<Derived>::Scalar l2(const Eigen::MatrixBase<Derived>& x) {
		return x.dot(x);
	}

}