#pragma once

#include "Creature.h"

namespace buglife {

	void Creature::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const {
		Object::draw(img, size, offset, scale, coeff);
		cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
		//cv::circle(img, p, radius * coeff * scale, color, -1);
		auto lcolor = cv::Scalar(std::min(255, color[0] + 50), std::min(255, color[1] + 50), std::min(255, color[2] + 50));
		//cv::circle(img, p, radius * coeff * scale, lcolor, 1);
		float r = radius * coeff * scale;
		cv::circle(img, p, (energy / maxEnergy) * r, lcolor, -1);
		cv::Point2f p1 = p + cv::Point2f(r * cos(orient), r * sin(orient));
		cv::Point2f p2 = p1 + coeff * scale * cv::Point2f(lastLook.dist * cos(orient + eyeOrient), lastLook.dist * sin(orient + eyeOrient));
		cv::line(img, p1, p2, lastLook.color);
		if (triesToBite) cv::circle(img, p1, coeff * scale * 0.1f, CV_RGB(255, 0, 0), -1);
		if (triesToLay) cv::circle(img, p, radius * coeff * scale, CV_RGB(50, 50, 255), 1);
	}

	void Creature::feed(float nutr) {
		energy += nutr;
		energy = std::clamp(energy, 0.0f, maxEnergy);
	}

	void Creature::bitten(float dmg) {
		energy -= dmg;
		destroyed = (energy <= 0.0f);
	}

	void Creature::bite(Object& obj) {
		float nutr = obj.getNutrition();
		float dmg = std::min(nutr, energy * BL_BITE_DMG);
		if (nutr > 0.0f)
			feed(dmg + maxEnergy * BL_BITE_DRAIN_PERCENT);
		obj.bitten(dmg);
	}

	Egg Creature::layEgg() {
		Species s(species); 
		if (BL_RAND_FLOAT < BL_MUT_PROB) s.mutate(BL_CRAZY_MUT_PROB);
		return Egg(s, pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f));
	}

	void Creature::live(const LookInfo& li, double dt) {
		prvEnergy = energy;
		isBiting = false;
		isLaying = false;
		lastLook = li;
		Eigen::Matrix<BL_SCALAR, 1, BL_IN_LAYER> awareness;
		awareness[0] = boost;
		awareness[1] = d_orient;
		awareness[2] = triesToBite ? 1.0f : 0.0f;
		awareness[3] = triesToLay ? 1.0f : 0.0f;
		awareness[4] = energy / maxEnergy;
		awareness[5] = (energy - prvEnergy) / maxEnergy;
		awareness[6] = 0.5f + eyeOrient / 3.14159f;
		awareness[7] = (li.dist > species.eyesight || li.dist < 0.0f) ? 1.0f : li.dist / species.eyesight;
		awareness[8] = float(li.color[0]) / 255.0f;
		awareness[9] = float(li.color[1]) / 255.0f;
		awareness[10] = float(li.color[2]) / 255.0f;
		
		Eigen::Matrix<BL_SCALAR, 1, BL_OUT_LAYER> reaction;
		species.brain.forward(awareness, &reaction);
		//std::cout << reaction << std::endl;

		boost = reaction[0];
		d_orient = (3.14159 * (reaction[1] - 0.5f));
		triesToBite = (reaction[2] > 0.75f);
		triesToLay = (reaction[3] > 0.75f);

		if (!triesToBite && !triesToLay) {
			if (abs(d_orient) > BL_ROT_THRESH) {
				orient += d_orient * dt;
				energy -= dt * d_orient * BL_WALK_DRAIN;
			}
			vel = boost * cv::Point2f(cos(orient), sin(orient));
			energy -= dt * boost * BL_WALK_DRAIN;
		}
		energy -= dt * BL_BASE_DRAIN;

		if (triesToLay) {
			vel *= 0.0f;
		}

		if (energy > 0.0f) {

			if (triesToLay && energy > maxEnergy * BL_BIRTH_DRAIN_PERCENT) {
				isLaying = true;
				energy -= maxEnergy * BL_BIRTH_DRAIN_PERCENT;
			}

			if (!triesToLay && triesToBite && energy > maxEnergy * BL_BITE_DRAIN_PERCENT) {
				isBiting = true;
				energy -= maxEnergy * BL_BITE_DRAIN_PERCENT;
			}
		} else {
			destroyed = true;
		}
	}
}