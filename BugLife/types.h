#pragma once

#include <set>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define BL_MAX_EYES 4
#define BL_SCALAR float
#define BL_IN_LAYER BL_MAX_EYES*4+1
#define BL_OUT_LAYER 3
#define BL_DEFAULT_DESC nnn::NNDesc({{{BL_IN_LAYER, nnn::ActivationFunctionType::NONE},{ BL_IN_LAYER, nnn::ActivationFunctionType::SIGMOID },{ BL_MAX_EYES*2, nnn::ActivationFunctionType::SIGMOID },{ BL_OUT_LAYER, nnn::ActivationFunctionType::SIGMOID }},nnn::OptimizerType::REGULAR,nnn::LossFunctionType::L2})
#define BL_MUT_PROB 0.9f

namespace buglife {

	class Object {
	public:
		cv::Point2f pos;
		cv::Vec3b color;
		float radius;

		Object(const cv::Vec3b& color, const cv::Point2f& pos, float radius) :
			color(color), pos(pos), radius(radius)
		{ }
	};

}