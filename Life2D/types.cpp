#include "types.h"

#include <algorithm>

namespace life2d {

	void PointMass::applyForce(cv::Point2f dir, float strength) {
		if (!fixed)
			acc += dir * strength;
	}

	void PointMass::update(double dt) {
		vel = pos - prv;
		prv = pos;
		pos += vel + acc * dt * dt;
		acc = {};
	}

	cv::Point2f closestPointOnLine(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p) {
		cv::Point2f line = p2 - p1;
		float line_length = std::sqrt(line.x * line.x + line.y * line.y);
		cv::Point2f line_direction = line / line_length;
		cv::Point2f p1_to_p = p - p1;
		float projection = p1_to_p.dot(line_direction);
		cv::Point2f closest_point = p1 + line_direction * projection;
		return closest_point;
	}

	void PointMass::collide(const Plane& p) {
		if (fixed || !collideable)
			return;
		auto ballpoint = (pos - p.normal * radius);
		bool inside = p.normal.dot(ballpoint - p.pos) < 0.0f;
		if (!inside)
			return;
		float angle = atan2(p.normal.y, p.normal.x);
		cv::Point2f p1 = p.pos + 1e2 * cv::Point2f(cos(angle + 3.14159f / 2.0f), sin(angle + 3.14159f / 2.0f));
		cv::Point2f p2 = p.pos + 1e2 * cv::Point2f(cos(angle - 3.14159f / 2.0f), sin(angle - 3.14159f / 2.0f));
		auto col = closestPointOnLine(p1, p2, pos);
		auto dist = cv::norm(col - pos);
		pos = col + p.normal * radius;
	}

	void PointMass::collide(PointMass& pm) {
		if (!collideable || !pm.collideable)
			return;
		if (fixed && pm.fixed)
			return;
		const float response_coef = 0.75f;
		auto v = pos - pm.pos;
		float dist2 = v.x * v.x + v.y * v.y;
		float min_dist = radius + pm.radius;
		if (dist2 < min_dist * min_dist) {
			const float dist = sqrt(dist2);
			auto n = v / dist;
			const float sumass = mass + pm.mass;
			const float mr1 = mass / sumass;
			const float mr2 = pm.mass / sumass;
			const float delta = 0.5f * response_coef * (dist - min_dist);
			if (!fixed) pos -= n * mr1 * delta;
			if (!pm.fixed) pm.pos += n * mr2 * delta;
		}
	}

	void PointMass::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) {
		cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
		cv::circle(img, p, radius * coeff * scale, cv::Scalar(255, 255, 255));
		cv::circle(img, p, 1, cv::Scalar(255, 255, 255), -1);
	}

	void PointMassLink::constrain(PointMass* pms) {

	}

	void PointMassLink::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, PointMass* pms) {
		cv::Point2f p1 = coeff * scale * pms[id1].pos + offset + (size / 2.0f);
		cv::Point2f p2 = coeff * scale * pms[id2].pos + offset + (size / 2.0f);
		float redcoeff = std::clamp((float)cv::norm(p1 - p2) / length - 1.0f, 0.0f, 1.0f);
		uchar nonred = floor((1.0f - fabs(redcoeff)) * 255.0f);
		int thickness = int(floor(2.5f + 2.5f * redcoeff));
		cv::line(img, p1, p2, CV_RGB(255, nonred, nonred), thickness);
	}

	void Plane::update(double dt) {

	}

	void Plane::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) {
		cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
		float angle = atan2(normal.y, normal.x);
		cv::Point2f p1 = p + 1e4 * cv::Point2f(cos(angle + 3.14159f / 2.0f), sin(angle + 3.14159f / 2.0f));
		cv::Point2f p2 = p + 1e4 * cv::Point2f(cos(angle - 3.14159f / 2.0f), sin(angle - 3.14159f / 2.0f));
		cv::line(img, p1, p2, CV_RGB(255, 255, 255), 2);
		cv::line(img, p1 - normal * 3.0f, p2 - normal * 3.0f, CV_RGB(255, 255, 255));
	}

	void Polygon::update(double dt) {

	}

	void Polygon::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, PointMass* pms) {
		auto lcolor = cv::Scalar(std::min(255, (int)color[0] + 50), std::min(255, (int)color[1] + 50), std::min(255, (int)color[2] + 50));
		auto dcolor = cv::Scalar(std::min(255, (int)color[0] + 50), std::min(255, (int)color[1] + 50), std::min(255, (int)color[2] + 50));
		cv::Mat temp = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		std::vector<cv::Point2f> pts;
		for (auto id : ids)
			pts.push_back(pms[id].pos);
		cv::polylines(temp, pts, true, dcolor, -1);
		cv::polylines(temp, pts, true, lcolor, 1);
		cv::addWeighted(img, 0.5, temp, 0.5, 1.0, img);
	}

}