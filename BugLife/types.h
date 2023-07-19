#pragma once

#include <set>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "../NNN/NeuralNet.hpp"

#define BL_RAND_FLOAT static_cast <float> (rand()) / (static_cast <float> (RAND_MAX))

// in
// prv(vel d_ori a_bite a_lay) nrg d_nrg e_ori closeness r g b  (11)
// out
// vel d_ori a_bite a_lay

#define BL_SCALAR float
#define BL_IN_LAYER 11
#define BL_OUT_LAYER 4
#define BL_DEFAULT_DESC \
	nnn::NNDesc({\
		{\
			{BL_IN_LAYER, nnn::ActivationFunctionType::NONE},\
			{BL_IN_LAYER, nnn::ActivationFunctionType::SIGMOID },\
			{BL_IN_LAYER, nnn::ActivationFunctionType::SIGMOID },\
			{8          , nnn::ActivationFunctionType::SIGMOID },\
			{8          , nnn::ActivationFunctionType::SIGMOID },\
            {BL_OUT_LAYER, nnn::ActivationFunctionType::SIGMOID }\
		}, nnn::OptimizerType::ADAM, nnn::LossFunctionType::L2\
	})
#define BL_CRAZY_MUT_PROB 0.02f
#define BL_BASE_DRAIN_PER_SECOND 0.1f

#define BL_EGG_RADIUS 0.2f
#define BL_MUT_PROB 0.45f
#define BL_BRAIN_W_MUT_PROB 0.03f
#define BL_DEF_EYESIGHT 10.0f
#define BL_MAX_EYESIGHT 50.0f
#define BL_WALK_DRAIN 0.05f
#define BL_BASE_DRAIN 0.01f
#define BL_BITE_DRAIN_PERCENT 0.05f
#define BL_BIRTH_DRAIN_PERCENT 0.7f
#define BL_BITE_DMG 0.5f
#define BL_MAX_FOODS 400
#define BL_FOOD_RADIUS 0.3f
#define BL_FOOD_COLOR cv::Vec3b(0, 255, 0)
#define BL_TIME_TO_HATCH 5.0f
#define BL_BITE_DIST 1.0f
#define BL_ROT_THRESH 0.05f

namespace buglife {

	class Object {
	public:
		cv::Point2f pos, prvPos, vel;
		cv::Vec3b color;
		float radius;
		bool dynamic;
        bool destroyed = false;        

		Object(const cv::Vec3b& color, cv::Point2f pos, float radius, bool dynamic) :
			color(color), pos(pos), prvPos(pos), radius(radius), dynamic(dynamic)
		{
			vel = { 0, 0 };
		}

        virtual float getNutrition() { return -1.0f; }
		virtual bool isFood() { return false; }
		virtual void bitten(float dmg) { return; }
		
        virtual void update(double dt);

        virtual void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const;
	};

	typedef Object Rock;

	struct Species {
		float radius = 1.0f;
		cv::Vec3b color = { 100, 100, 200 };
		cv::Vec3b eggColor = { 100, 100, 200 };
		float eyesight = BL_DEF_EYESIGHT;
		nnn::NeuralNet<BL_SCALAR, BL_IN_LAYER, BL_OUT_LAYER> brain;

		Species() : 
			brain(BL_DEFAULT_DESC)
		{
			brain.initialize(0.0f, 1.0f);
		}

		Species(const Species& s) :
			radius(s.radius),
			color(s.color),
			eggColor(s.eggColor),
			eyesight(s.eyesight),
			brain(s.brain)
		{ }

		float getMaxEnergy() const;
		void mutateColor(cv::Vec3b& c);
		void mutate(float crazyMutProb);
	};

	bool intersectCircleBySegment(const cv::Point2f& s, const cv::Point2f& e, const cv::Point2f& o, float r, cv::Point2f& point);

}