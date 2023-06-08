#pragma once

#include <set>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define BL_MAX_EYES 4
#define BL_SCALAR float
#define BL_IN_LAYER BL_MAX_EYES*4+1
#define BL_OUT_LAYER 3
#define BL_DEFAULT_DESC nnn::NNDesc({{{BL_IN_LAYER, nnn::ActivationFunctionType::NONE},{ BL_IN_LAYER, nnn::ActivationFunctionType::SIGMOID },{ BL_MAX_EYES*2, nnn::ActivationFunctionType::SIGMOID },{ BL_OUT_LAYER, nnn::ActivationFunctionType::SIGMOID }},nnn::OptimizerType::SIMPLE,nnn::LossFunctionType::L2})
#define BL_MUT_PROB 0.9f
#define BL_BASE_DRAIN_PER_SECOND 0.1f

namespace buglife {

	class Object {
	public:
		cv::Point2f pos;
		cv::Vec3b color;
		float radius;

		Object(const cv::Vec3b& color, const cv::Point2f& pos, float radius) :
			color(color), pos(pos), radius(radius)
		{ }
		
		virtual void update(double time) 
		{ }

		virtual void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const {
			cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
			cv::circle(img, p, radius * coeff * scale, color, -1);
			auto lcolor = cv::Scalar(std::min(255, color[0] + 50), std::min(255, color[1] + 50), std::min(255, color[2] + 50));
			cv::circle(img, p, radius * coeff * scale, lcolor, 1);
		}
	};

}