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
		Object* target = nullptr;
	};

	class Creature : public Object {
	public:
		float orient, d_orient, eyeOrient;
		float boost = 0;
		float maxEnergy;
		float energy, prvEnergy, drain;
		bool triesToBite, isBiting;
		bool triesToLay, isLaying;
		bool triesToTarget;
		Object* target = nullptr;
		float tOrient;
		Species species;
		LookInfo lastLook;
		float age = 0.0f;
		float lastTemp = BL_CREATURE_TEMP, lastFwdTemp = BL_CREATURE_TEMP;

		Creature() { }

		Creature(const Species& species, const cv::Point2f& pos, float orient) :
			Object(species.color, pos, species.radius, true),
			species(species),
			orient(orient),
			maxEnergy(species.getMaxEnergy()),
			energy(maxEnergy / 2.0f),
			prvEnergy(maxEnergy / 2.0f),
			drain(maxEnergy / (2.0f * 3.14159)),
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

		float getNutrition() { return maxEnergy / 3.0f; }

		void live(const LookInfo& li, double dt, const cv::Size2i& wsz);
		void handleTemperature(float temp, float fwdTemp);

		virtual void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const;

		void feed(float nutr);
		void bitten(float dmg);
		void bite(Object& obj);
		
		bool tryLay();
		Egg layEgg(float mutProb = BL_MUT_PROB);

		void save(std::ofstream& out) {
			Object::save(out);
			out.write(reinterpret_cast<const char*>(&orient), sizeof(float));
			out.write(reinterpret_cast<const char*>(&d_orient), sizeof(float));
			out.write(reinterpret_cast<const char*>(&eyeOrient), sizeof(float));
			out.write(reinterpret_cast<const char*>(&boost), sizeof(float));
			out.write(reinterpret_cast<const char*>(&maxEnergy), sizeof(float));
			out.write(reinterpret_cast<const char*>(&energy), sizeof(float));
			out.write(reinterpret_cast<const char*>(&prvEnergy), sizeof(float));
			out.write(reinterpret_cast<const char*>(&drain), sizeof(float));
			species.save(out);
		}

		void load(std::ifstream& in) {
			Object::load(in);
			in.read(reinterpret_cast<char*>(&orient), sizeof(float));
			in.read(reinterpret_cast<char*>(&d_orient), sizeof(float));
			in.read(reinterpret_cast<char*>(&eyeOrient), sizeof(float));
			in.read(reinterpret_cast<char*>(&boost), sizeof(float));
			in.read(reinterpret_cast<char*>(&maxEnergy), sizeof(float));
			in.read(reinterpret_cast<char*>(&energy), sizeof(float));
			in.read(reinterpret_cast<char*>(&prvEnergy), sizeof(float));
			in.read(reinterpret_cast<char*>(&drain), sizeof(float));
			species.load(in);
		}
	};

}