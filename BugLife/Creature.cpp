#pragma once

#include "Creature.h"

namespace buglife {

	void Creature::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const {
		Object::draw(img, size, offset, scale, coeff);
		cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
		auto lcolor = cv::Scalar(std::min(255, color[0] + 50), std::min(255, color[1] + 50), std::min(255, color[2] + 50));
		float r = radius * coeff * scale;
		cv::circle(img, p, (std::max(energy, 0.0f) / maxEnergy) * r, lcolor, -1);
		cv::Point2f p1 = p + cv::Point2f(r * cos(orient), r * sin(orient));
		cv::Point2f p2 = p1 + coeff * scale * cv::Point2f(lastLook.dist * cos(orient + eyeOrient), lastLook.dist * sin(orient + eyeOrient));
		cv::Point2f p3 = p1 + coeff * scale * cv::Point2f(0.5 * cos(orient + eyeOrient), 0.5 * sin(orient + eyeOrient));
		cv::line(img, p1, p3, lastLook.color);
		if (triesToBite) cv::circle(img, p1, coeff * scale * 0.1f, CV_RGB(255, 0, 0), -1);
		if (triesToLay) cv::circle(img, p, radius * coeff * scale, CV_RGB(50, 50, 255), 1);
		if (triesToTarget) cv::line(img, p1, p2, lastLook.color);
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
		if (obj.isFood())
			feed(dmg + maxEnergy * BL_BITE_DRAIN_PERCENT);
		obj.bitten(dmg);
	}

	bool Creature::tryLay() {
		energy -= maxEnergy * BL_TRY_BIRTH_DRAIN_PERCENT / 2.0f;
		float r = BL_RAND_FLOAT;
		return r < (age / BL_MAX_AGE);
	}

	Egg Creature::layEgg(float mutProb) {
		Species s(species); 
		energy -= maxEnergy * BL_BIRTH_DRAIN_PERCENT / 2.0f;
		if (BL_RAND_FLOAT < mutProb) s.mutate(BL_CRAZY_MUT_PROB);
		return Egg(s, pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f));
	}

	void Creature::live(const LookInfo& li, double dt, const cv::Size2i& wsz) {

		if (target && !target->destroyed) {
			auto tpos = target->pos - pos;
			if (tpos.x < -wsz.width / 2) tpos.x += wsz.width;
			if (tpos.x > wsz.width / 2) tpos.x -= wsz.width;
			if (tpos.y < -wsz.height / 2) tpos.y += wsz.height;
			if (tpos.y > wsz.height / 2) tpos.y -= wsz.height;
			float ori = atan2(tpos.y, tpos.x);
			tOrient = ori - orient;
			if (cv::norm(tpos) > species.eyesight || abs(tOrient) > 3.14159f / 2.0f) {
				target = nullptr;
			}
		}

		prvEnergy = energy;
		isBiting = false;
		isLaying = false;
		lastLook = li;
		Eigen::Matrix<BL_SCALAR, 1, BL_IN_LAYER> awareness;
		awareness[0] = energy / maxEnergy;
		awareness[1] = (energy - prvEnergy) / maxEnergy;
		awareness[2] = (li.dist > species.eyesight || li.dist < 0.0f) ? 1.0f : li.dist / species.eyesight;
		awareness[3] = float(li.color[0]) / 255.0f;
		awareness[4] = float(li.color[1]) / 255.0f;
		awareness[5] = float(li.color[2]) / 255.0f;
		awareness[6] = target ? 1.0f : 0.0f;
		awareness[7] = target ? (0.5f + tOrient / 3.14159f) : 0.5f;
		awareness[8] = (lastTemp < BL_CREATURE_TEMP) ? (0.5f * lastTemp / BL_CREATURE_TEMP) : (0.5f + 0.5f * std::clamp(lastTemp / 255.0f, 0.0f, 1.0f));
		awareness[9] = (lastTemp > lastFwdTemp) ? 0.0f : 1.0f;
		awareness[10] = std::clamp(age / BL_MAX_AGE, 0.0f, 1.0f);
		
		Eigen::Matrix<BL_SCALAR, 1, BL_OUT_LAYER> reaction;
		species.brain.forward(awareness, &reaction);

		boost = 2.0f * (reaction[0] - 0.5f);
		d_orient = (3.14159f * (reaction[1] - 0.5f));
		triesToBite = (reaction[2] > 0.75f);
		triesToLay = (reaction[3] > 0.75f);
		triesToTarget = (reaction[4] > 0.75f);

		if (!triesToBite && !triesToLay) {
			if (abs(d_orient) > BL_ROT_THRESH) {
				orient += d_orient * dt;
				energy -= dt * abs(d_orient) * BL_ROT_DRAIN * drain * BL_DRAIN_COEFF;
			}
			vel = boost * cv::Point2f(cos(orient), sin(orient));
			energy -= dt * abs(boost) * BL_WALK_DRAIN * drain * BL_DRAIN_COEFF;
		}
		while (orient > 3.14159f) orient -= 2.0f * 3.14159f;
		while (orient < -3.14159f) orient += 2.0f * 3.14159f;
		energy -= dt * BL_BASE_DRAIN * drain * BL_DRAIN_COEFF;

		if (triesToLay) {
			vel *= 0.0f;
		}

		if (energy > 0.0f) {

			if (triesToLay && energy > maxEnergy * BL_BIRTH_DRAIN_PERCENT) {
				isLaying = true;
			}

			if (!triesToLay && triesToBite && energy > maxEnergy * BL_BITE_DRAIN_PERCENT) {
				isBiting = true;
				energy -= maxEnergy * BL_BITE_DRAIN_PERCENT;
			}

			energy = std::clamp(energy, 0.0f, maxEnergy);
		} else {
			destroyed = true;
		}

		if (triesToTarget && lastLook.target) {
			target = lastLook.target;
		}

		age += dt;
		drain = ((maxEnergy / (2.0f * 3.14159)) * (1.0f + (age / (radius * BL_MAX_AGE))) + abs(lastTemp - BL_CREATURE_TEMP) * BL_BASE_TEMP_DRAIN);
	}

	void Creature::handleTemperature(float temp, float fwdTemp) {
		lastTemp = temp;
		lastFwdTemp = fwdTemp;
	}
}