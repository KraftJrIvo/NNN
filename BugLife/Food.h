#pragma once

#include "types.h"

namespace buglife {

	class Food : public Object {
	public:
		float nutrition;

		Food() { }

		Food(const cv::Point2f& pos, float radius = BL_FOOD_RADIUS, cv::Vec3b color = BL_FOOD_COLOR) :
			Object(color, pos, radius, true),
			nutrition(2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS)
		{ }

		float getNutrition() { return nutrition; }
		bool isFood() { return true; }
		void bitten(float dmg) { destroyed = true; }

		virtual void save(std::ofstream& out) {
			Object::save(out);
			out.write(reinterpret_cast<const char*>(&nutrition), sizeof(float));
		}

		virtual void load(std::ifstream& in) {
			Object::load(in);
			in.read(reinterpret_cast<char*>(&nutrition), sizeof(float));
		}
	};

	class Poison : public Food {
	public:
		Poison() { }
		Poison(const cv::Point2f& pos, float radius = BL_POISON_RADIUS, cv::Vec3b color = BL_POISON_COLOR) :
			Food(pos, radius, color)
		{
			nutrition *= -100000.0f;
		}
	};

	class Egg : public Food {
	public:
		float timeToHatch;
		bool hatching = false;
		Species species;

		Egg() { }

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

		virtual void save(std::ofstream& out) {
			Food::save(out);
			species.save(out);
		}

		virtual void load(std::ifstream& in) {
			Food::load(in);
			species.load(in);
		}
	};
}