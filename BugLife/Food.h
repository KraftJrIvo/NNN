#pragma once

#include "Creature.h"

namespace buglife {

	class Food : public Object {
	public:
		float nutrition;

		Food(const cv::Vec3b& color, const cv::Point2f& pos, float radius, float nutrition) :
			Object(color, pos, radius),
			nutrition(nutrition)
		{ }

	};

	class Egg : public Food {
	public:
		float nutrition;

		Egg(const cv::Vec3b& color, const cv::Point2f& pos, float radius, float nutrition, const Creature& creature) :
			Food(color, pos, radius, nutrition),
			_creature(creature)
		{ }

	private:
		Creature _creature;

	};
}