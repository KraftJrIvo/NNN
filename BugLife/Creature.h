#pragma once

#include <algorithm>

#include "types.h"
#include "Food.h"
#include "../NNN/NeuralNet.hpp"

namespace buglife {

	struct LookInfo {
		bool seeSmth = false;
		float dist = -1.0f;
		cv::Vec3b color = { 100, 100, 100 };
		Object* target;
	};

	class Creature : public Object {
	public:
		float orient, d_orient, eyeOrient;
		float boost = 0;
		float maxEnergy;
		float energy, prvEnergy;
		bool triesToBite, isBiting;
		bool triesToLay, isLaying;
		Species species;
		LookInfo lastLook;

		Creature(const Species& species, const cv::Point2f& pos, float orient) :
			Object(species.color, pos, species.radius, true),
			species(species),
			orient(orient),
			maxEnergy(species.getMaxEnergy()),
			energy(maxEnergy / 2.0f),
			prvEnergy(maxEnergy / 2.0f),
			eyeOrient(0)
		{ }

		Creature(const Egg& egg) :
			Object(egg.species.color, egg.pos, egg.species.radius, true),
			species(egg.species),
			orient(BL_RAND_FLOAT * 2.0f * 3.14159),
			maxEnergy(egg.species.getMaxEnergy()),
			energy(maxEnergy),
			prvEnergy(maxEnergy),
			eyeOrient(0)
		{ }

		Creature(const Creature& cre) :
			Creature(cre.species, cre.pos, cre.orient)
		{ }

		float getNutrition() { return maxEnergy / 2.0f; }

		void live(const LookInfo& li, double dt);

		virtual void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const;

		void feed(float nutr);
		void bitten(float dmg);
		void bite(Object& obj);
		
		Egg layEgg();
	};

}