#pragma once

#include "types.h"
#include "../NNN/NeuralNet.hpp"

namespace buglife {

	class Creature : public Object {
	public:
		float orient;
		int maxEnergy, energy;

		Creature(const cv::Vec3b& color, const cv::Point2f& pos, float radius = 3.0f) :
			Object(color, pos, radius),
			_nn(BL_DEFAULT_DESC)
		{ }

	private:
		std::set<Object> _objects;
		nnn::NeuralNet<BL_SCALAR, BL_IN_LAYER, BL_OUT_LAYER> _nn;
	};

}