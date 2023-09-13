#include "World.h"

#include <chrono>

#define SUB_STEPS 4

namespace life2d {

	void World::generate() {

		Plane p1 = { { size.width / 2.0f, 9.0f * size.height / 10.0f }, { 0.0f, -1.0f } };
		Plane p2 = { { size.width / 4.0f, 9.0f * size.height / 10.0f }, { (float)sqrt(2) / 2.0f, -(float)sqrt(2) / 2.0f}};
		Plane p3 = { { 3.0f * size.width / 4.0f, 9.0f * size.height / 10.0f }, { -(float)sqrt(2) / 2.0f, -(float)sqrt(2) / 2.0f} };
		_addPlane(p1); _addPlane(p2); _addPlane(p3);

		PointMass pm1 = { { size.width / 2.0f, size.height / 2.0f }, 5.0f, true, true };
		_addPointMass(pm1);

		for (int i = 0; i < 100; ++i) {
			float r1 = L2D_RAND_FLOAT * 2.0f - 1.0;
			float r2 = L2D_RAND_FLOAT * 2.0f - 1.0;
			float r3 = L2D_RAND_FLOAT * 2.9f + 0.1;
			cv::Point2f pt = { size.width / 2.0f + r1, size.height / 2.0f - r2 };
			_addPointMass({ pt, r3, false, true });
		}
	}

	size_t World::_addPlane(const Plane& p) {
		_planes.push_back(p);
		return _planes.size() - 1;
	}

	size_t World::_addPointMass(const PointMass& pm) {
		_pointMasses.push_back(pm);
		return _pointMasses.size() - 1;
	}

	size_t World::_addLink(const PointMassLink& l) {
		_links.push_back(l);
		return _links.size() - 1;
	}

	size_t World::_addPolygon(const Polygon& p) {
		_polygons.push_back(p);
		return _polygons.size() - 1;
	}

	void World::update() {
		std::chrono::duration<double> seconds = std::chrono::system_clock::now() - _startTime;
		auto dt = (seconds.count() - _time);
		_time = seconds.count();

		lock();
		for (int s = 0; s < SUB_STEPS; ++s) {
			for (int i = 0; i < _pointMasses.size(); ++i) {
				auto& p1 = _pointMasses[i];
				p1.applyForce({ 0.0f, 1.0f }, 300.0f);
				for (int j = 0; j < _planes.size(); ++j) {
					auto& p = _planes[j];
					p1.collide(p);
				}
				for (int j = 0; j < _pointMasses.size(); ++j) {
					if (i == j) continue;
					auto& p2 = _pointMasses[j];
					p1.collide(p2);
				}
				p1.update(dt / SUB_STEPS);
			}
			//for (int i = 0; i < _pointMasses.size(); ++i) {
			//	auto& p1 = _pointMasses[i];
			//	p1.update(dt / SUB_STEPS);
			//}
		}
		unlock();
	}
}