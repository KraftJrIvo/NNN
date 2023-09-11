#pragma once

#include <set>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "../NNN/NeuralNet.hpp"

#define BL_RAND_FLOAT static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) + 1.0f)

// in
// nrg d_nrg closeness r g b target t_ori  (11)
// out
// vel d_ori a_bite a_lay a_target (5)

#define BL_SCALAR float
#define BL_IN_LAYER 11
#define BL_OUT_LAYER 5
#define BL_DEFAULT_DESC \
	nnn::NNDesc({\
		{\
			{BL_IN_LAYER, nnn::ActivationFunctionType::NONE},\
			{16, nnn::ActivationFunctionType::SIGMOID },\
			{14, nnn::ActivationFunctionType::SIGMOID },\
			{10, nnn::ActivationFunctionType::SIGMOID },\
			{8, nnn::ActivationFunctionType::SIGMOID },\
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
#define BL_WALK_DRAIN 0.03f
#define BL_ROT_DRAIN 0.02f
#define BL_BASE_DRAIN 0.005f
#define BL_BASE_TEMP_DRAIN 100.0f
#define BL_BITE_DRAIN_PERCENT 0.05f
#define BL_BIRTH_DRAIN_PERCENT 0.7f
#define BL_TRY_BIRTH_DRAIN_PERCENT 0.1f
#define BL_BITE_DMG 0.5f
#define BL_DRAIN_COEFF 1.0f

#define BL_FOOD_RADIUS 0.3f
#define BL_FOOD_COLOR cv::Vec3b(0, 255, 0)
#define BL_FOOD_EXPIRY 300.0f
#define BL_MAX_FOODS 125
#define BL_POISON_RADIUS 0.3f
#define BL_POISON_COLOR cv::Vec3b(0, 0, 255)
#define BL_MAX_POISON 50

#define BL_TIME_TO_HATCH 5.0f
#define BL_BITE_DIST 0.5f
#define BL_ROT_THRESH 0.05f
#define BL_MAX_CREATURES 200
#define BL_MAX_AGE 1000.0f
#define BL_AUTOSAVE_INTERVAL 60.0f
#define BL_TIMESCALE 15.0f

#define BL_TEMP_RES 3.0f
#define BL_TEMP_DIFFUSIVITY 0.05f
#define BL_CREATURE_TEMP 0.1f
#define BL_NIGHT_D_TEMP -0.05f
#define BL_DAY_D_TEMP 0.025f
#define BL_DAY_LEN 60.0f

namespace buglife {

	class Object {
	public:
		cv::Point2f pos, prvPos, vel;
		cv::Vec3b color;
		float radius;
		bool dynamic;
        bool destroyed = false;  

		Object() { }

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

		virtual void save(std::ofstream& out) {
			out.write(reinterpret_cast<const char*>(&pos.x), sizeof(float));
			out.write(reinterpret_cast<const char*>(&pos.y), sizeof(float));
			out.write(reinterpret_cast<const char*>(&prvPos.x), sizeof(float));
			out.write(reinterpret_cast<const char*>(&prvPos.y), sizeof(float));
			out.write(reinterpret_cast<const char*>(&vel.x), sizeof(float));
			out.write(reinterpret_cast<const char*>(&vel.y), sizeof(float));
			out.write(reinterpret_cast<const char*>(&color[0]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&color[1]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&color[2]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&radius), sizeof(float));
			out.write(reinterpret_cast<const char*>(&dynamic), sizeof(bool));
			out.write(reinterpret_cast<const char*>(&destroyed), sizeof(bool));
		}

		virtual void load(std::ifstream& in) {
			in.read(reinterpret_cast<char*>(&pos.x), sizeof(float));
			in.read(reinterpret_cast<char*>(&pos.y), sizeof(float));
			in.read(reinterpret_cast<char*>(&prvPos.x), sizeof(float));
			in.read(reinterpret_cast<char*>(&prvPos.y), sizeof(float));
			in.read(reinterpret_cast<char*>(&vel.x), sizeof(float));
			in.read(reinterpret_cast<char*>(&vel.y), sizeof(float));
			in.read(reinterpret_cast<char*>(&color[0]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&color[1]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&color[2]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&radius), sizeof(float));
			in.read(reinterpret_cast<char*>(&dynamic), sizeof(bool));
			in.read(reinterpret_cast<char*>(&destroyed), sizeof(bool));
		}
	};

	typedef Object Rock;
	typedef Object Tree;

	struct Species {
		float radius = 0.5f;
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

		void save(std::ofstream& out) {
			out.write(reinterpret_cast<const char*>(&radius), sizeof(float));
			out.write(reinterpret_cast<const char*>(&color[0]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&color[1]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&color[2]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&eggColor[0]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&eggColor[1]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&eggColor[2]), sizeof(uchar));
			out.write(reinterpret_cast<const char*>(&eyesight), sizeof(float));
			brain.save(out);
		}

		void save(const std::string& filePath) {
			std::ofstream out(filePath, std::ios::binary);
			save(out);
		}

		void load(std::ifstream& in) {
			in.read(reinterpret_cast<char*>(&radius), sizeof(float));
			in.read(reinterpret_cast<char*>(&color[0]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&color[1]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&color[2]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&eggColor[0]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&eggColor[1]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&eggColor[2]), sizeof(uchar));
			in.read(reinterpret_cast<char*>(&eyesight), sizeof(float));
			brain.load(in);
		}

		void load(const std::string& filePath) {
			std::ifstream in(filePath, std::ios::binary);
			load(in);
		}
	};

	bool intersectCircleBySegment(const cv::Point2f& s, const cv::Point2f& e, const cv::Point2f& o, float r, cv::Point2f& point);

}