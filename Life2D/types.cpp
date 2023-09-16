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
		acc = { 0.0f, 0.0f };
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

		auto fricoeff = friction * p.friction;
		auto dir = p2 - pos;
		auto paral = 2.0f * (dir / cv::norm(dir)) * (vel.dot(dir) / cv::norm(dir));
		
		pos += 0.5f * ((col + p.normal * radius - pos) - paral * fricoeff);
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
			const float mr1 = fixed ? 1.0f : pm.fixed ? 0.0f : (mass / sumass);
			const float mr2 = 1.0f - mr1;
			const float delta = 0.5f * response_coef * (dist - min_dist);
			if (!fixed) pos -= n * mr2 * delta;
			if (!pm.fixed) pm.pos += n * mr1 * delta;
		}
	}

	void PointMass::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, bool selected) {
		cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
		auto col = selected ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 255);
		cv::circle(img, p, radius * coeff * scale, col);
		cv::circle(img, p, 1, col, -1);
	}

	void PointMassLink::constrain(PointMass* pms, double dt) {
		if (id1 == id2)
			return;
		auto& pm1 = pms[id1];
		auto& pm2 = pms[id2];
		const float sumass = pm1.mass + pm2.mass;
		const float mr1 = pm1.fixed ? 1.0f : pm2.fixed ? 0.0f : (pm1.mass / sumass);
		const float mr2 = 1.0f - mr1;
		
		auto d = pm2.pos - pm1.pos;
		auto dist = cv::norm(d);
		auto dir = d / dist;
		auto v1 = pm2.pos - dir * length;
		auto v2 = pm1.pos + dir * length;

		float stifcoeff = (stiffness < 1.0f) ? (stiffness * dt * 10.0f) : 1.0f;
		float dampcoeff = (damping < 1.0f) ? (damping * dt * 25.0f) : 0.1f;
		if (!pm1.fixed) pm1.pos += 0.5f * ((v1 - pm1.pos) * mr2 * stifcoeff + (pm2.vel - pm1.vel) * dampcoeff);
		if (!pm2.fixed) pm2.pos += 0.5f * ((v2 - pm2.pos) * mr1 * stifcoeff + (pm1.vel - pm2.vel) * dampcoeff);
	}

	void PointMassLink::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff, PointMass* pms) {
		if (length == 0.0f)
			return;
		cv::Point2f p1 = coeff * scale * pms[id1].pos + offset + (size / 2.0f);
		cv::Point2f p2 = coeff * scale * pms[id2].pos + offset + (size / 2.0f);
		float redcoeff = std::clamp((float)cv::norm(pms[id1].pos - pms[id2].pos) / length, 0.0f, 2.0f) - 1.0f;
		uchar nonred = floor((1.0f - fabs(redcoeff)) * 50.0f);
		int thickness = 1 + int(floor(2.0f - 2.0f * redcoeff));
		cv::line(img, p1, p2, CV_RGB(50.0f + fabs(redcoeff) * 205.0f, nonred, nonred), thickness);
		if (collideable) {
			auto dir = pms[id2].pos - pms[id1].pos;
			auto angle = atan2(dir.y, dir.x) - 3.14159f / 2.0f;
			p1 = coeff * scale * (pms[id1].pos + pms[id1].radius * cv::Point2f(cos(angle), sin(angle))) + offset + (size / 2.0f);
			p2 = coeff * scale * (pms[id2].pos + pms[id2].radius * cv::Point2f(cos(angle), sin(angle))) + offset + (size / 2.0f);
			cv::line(img, p1, p2, CV_RGB(255, 255, 255));
			angle = atan2(dir.y, dir.x) + 3.14159f / 2.0f;
			p1 = coeff * scale * (pms[id1].pos + pms[id1].radius * cv::Point2f(cos(angle), sin(angle))) + offset + (size / 2.0f);
			p2 = coeff * scale * (pms[id2].pos + pms[id2].radius * cv::Point2f(cos(angle), sin(angle))) + offset + (size / 2.0f);
			cv::line(img, p1, p2, CV_RGB(255, 255, 255));
		}
	}

	void PointMassLink::collide(PointMass& pm, PointMass* pms) {
		if (!collideable || !pm.collideable)
			return;
		auto& pm1 = pms[id1]; 
		auto& pm2 = pms[id2];
		if (&pm1 == &pm || (&pm2 == &pm))
			return;
		if (pm1.fixed && pm2.fixed && pm.fixed)
			return;
		float sumass = (pm1.mass + pm2.mass);
		const float mr1 = pm1.fixed ? 1.0f : pm2.fixed ? 0.0f : (pm1.mass / sumass);
		auto mr2 = 1.0f - mr1;
		auto d = (pm2.pos - pm1.pos);
		auto l = cv::norm(d);
		auto dir = d / l;
		auto angle = atan2(dir.y, dir.x) - 3.14159f / 2.0f;
		auto c = (pms[id1].pos + d * 0.5);
		auto d2 = pm.pos - c;
		auto c_dist = cv::norm((pm.pos - pm.radius * d2 / cv::norm(d2)) - c);
		if (c_dist < l * 0.5f) {
			auto col = closestPointOnLine(pm1.pos, pm2.pos, pm.pos);
			auto progr = cv::norm(col - pm1.pos) / l;
			auto radius = pm1.radius + (pm2.radius - pm1.radius) * progr;
			const float response_coef = 0.75f;
			auto v = col - pm.pos;
			float dist2 = v.x * v.x + v.y * v.y;
			float min_dist = radius + pm.radius;
			if (dist2 < min_dist * min_dist) {
				const float dist = sqrt(dist2);
				auto n = v / dist;
				float sumass2 = sumass + pm.mass;
				float mr1_ = (pm1.fixed && pm2.fixed) ? 1.0f : pm.fixed ? 0.0f : (sumass / sumass2);
				float mr2_ = 1.0f - mr1;
				const float delta = 0.5f * response_coef * (dist - min_dist);

				auto fricoeff = friction * pm.friction;
				auto relvel = pm.vel - (pm1.vel + (pm2.vel - pm1.vel) * progr);
				auto paral = 2.0f * dir * (relvel.dot(d) / l);

				if (!pm1.fixed || !pm2.fixed) {
					pm1.pos -= (n * mr2 * mr2_ * (1.0f - progr)) * delta;
					pm2.pos -= (n * mr1 * mr2_ * progr) * delta;
				}
				if (!pm.fixed) pm.pos += n * mr1_ * delta - 0.5f * mr1_ * paral * fricoeff;
			}
		}
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