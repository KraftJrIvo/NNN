#pragma once

#include "types.hpp"

namespace nnn::loss {

	template<typename Derived>
	typename Eigen::MatrixBase<Derived>::Scalar square(const Eigen::MatrixBase<Derived>& x) {
		return x.dot(x);
	}

}