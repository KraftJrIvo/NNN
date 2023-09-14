#pragma once

#include <opencv2/opencv.hpp>

#define L2D_RAND_FLOAT static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) + 1.0f)

namespace life2d {

	struct Plane {
		cv::Point2f pos;
		cv::Point2f normal;

		void update(double dt);
		void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff);
	};

	struct PointMass {
		cv::Point2f pos;
		float radius, mass;
		bool fixed, collideable;
		cv::Point2f prv, vel, acc;

		PointMass(const cv::Point2f& pos, float radius, bool fixed, bool collideable) : 
			pos(pos), radius(radius), fixed(fixed), collideable(collideable),
			prv(pos), vel({0.0f, 0.0f}), acc({0.0f, 0.0f}), mass(3.14159f * radius * radius)
		{ }

		void applyForce(cv::Point2f dir, float strength);
		void update(double dt);
		void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, bool selected = false);

		void collide(const Plane& p);
		void collide(PointMass& pm);
	};

	struct PointMassLink {
		size_t id1, id2;
		float length;
		float stiffness;
		float dampening;

		void constrain(PointMass* pms, double dt);
		void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, PointMass* pms);
	};

	struct Polygon {
		std::vector<size_t> ids;
		cv::Scalar color;
		float alpha;

		void update(double dt);
		void draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, PointMass* pms);
	};
}