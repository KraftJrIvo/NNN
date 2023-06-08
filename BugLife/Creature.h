#pragma once

#include "types.h"
#include "../NNN/NeuralNet.hpp"

namespace buglife {

	class Creature : public Object {
	public:
		float orient;
		float maxEnergy, energy;

		Creature(const nnn::NeuralNet<BL_SCALAR, BL_IN_LAYER, BL_OUT_LAYER>& nn, const cv::Vec3b& color, const cv::Point2f& pos, float radius = 3.0f) :
			Object(color, pos, radius),
			_nn(nn)
		{ }

		virtual void update(double time);

		float getWeight();
		float drainEnergy();

	private:
		nnn::NeuralNet<BL_SCALAR, BL_IN_LAYER, BL_OUT_LAYER> _nn;

	};

}