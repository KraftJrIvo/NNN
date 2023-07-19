#pragma once

#include "types.h"

namespace buglife {

	class Food : public Object {
	public:
		float nutrition;

		Food(const cv::Point2f& pos, float radius = BL_FOOD_RADIUS, cv::Vec3b color = BL_FOOD_COLOR) :
			Object(color, pos, radius, true),
			nutrition(2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS)
		{ }

		float getNutrition() { return nutrition; }
		bool isFood() { return true; }
		void bitten(float dmg) { destroyed = true; }
	};

	class Egg : public Food {
	public:
		float timeToHatch;
		bool hatching = false;
		Species species;

		Egg(const Species& species, const cv::Point2f& pos, float timeToHatch = BL_TIME_TO_HATCH) :
			Food(pos, BL_EGG_RADIUS, species.eggColor),
			species(species),
			timeToHatch(timeToHatch)
		{ }

		bool isHatching() { return hatching; }

		void update(double dt) {
			timeToHatch -= dt;
			hatching = (timeToHatch < 0.0f);
		}
	};
}