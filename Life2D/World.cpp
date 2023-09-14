#include "World.h"

#include <chrono>

#define SUB_STEPS 4

namespace life2d {

	void World::generate() {

		Plane p1 = { { size.width / 2.0f, 9.0f * size.height / 10.0f }, { 0.0f, -1.0f } };
		Plane p2 = { { size.width / 4.0f, 9.0f * size.height / 10.0f }, { (float)sqrt(2) / 2.0f, -(float)sqrt(2) / 2.0f}};
		Plane p3 = { { 3.0f * size.width / 4.0f, 9.0f * size.height / 10.0f }, { -(float)sqrt(2) / 2.0f, -(float)sqrt(2) / 2.0f} };
		_addPlane(p1); _addPlane(p2); _addPlane(p3);

		PointMass pm1 = { { size.width / 2.0f, size.height / 2.0f }, 0.1f, true, false };
		_addPointMass(pm1);
		_addLink({ 0, 0, 0.0f, 0.0f, 0.0f });

		_addPointMass({ {0.0f, 0.0f}, 1.0f, false, true});
		_addPointMass({ {1.0f, 0.0f}, 1.0f, false, true });
		_addPointMass({ {1.0f, 1.0f}, 1.0f, false, true });
		_addPointMass({ {0.0f, 1.0f}, 1.0f, false, true });
		_addLink({ 1, 2, 10.0f, 1.0f, 0.0f });
		_addLink({ 2, 3, 10.0f, 1.0f, 0.0f });
		_addLink({ 3, 4, 10.0f, 1.0f, 0.0f });
		_addLink({ 4, 1, 10.0f, 1.0f, 0.0f });
		_addLink({ 1, 3, sqrt(2.0f) * 10.0f, 1.0f, 0.0f });
		_addLink({ 2, 4, sqrt(2.0f) * 10.0f, 1.0f, 0.0f });
		
		_addPointMass({ {0.0f, 0.0f}, 1.0f, false, true });
		_addPointMass({ {1.0f, 0.0f}, 1.0f, false, true });
		_addPointMass({ {1.0f, 1.0f}, 1.0f, false, true });
		_addPointMass({ {0.0f, 1.0f}, 1.0f, false, true });
		_addLink({ 5, 6, 3.0f, 0.5f, 0.0f });
		_addLink({ 6, 7, 3.0f, 0.5f, 0.0f });
		_addLink({ 7, 8, 3.0f, 0.5f, 0.0f });
		_addLink({ 8, 5, 3.0f, 0.5f, 0.0f });
		_addLink({ 5, 7, sqrt(2.0f) * 3.0f, 0.5f, 0.0f });
		_addLink({ 6, 8, sqrt(2.0f) * 3.0f, 0.5f, 0.0f });

		for (int i = 0; i < 100; ++i) {
			float r1 = L2D_RAND_FLOAT * 2.0f - 1.0;
			float r2 = L2D_RAND_FLOAT * 2.0f - 1.0;
			float r3 = L2D_RAND_FLOAT * 2.9f + 0.1;
			cv::Point2f pt = { size.width / 2.0f + r1 * 50.0f, r2 * 50.0f };
			_addPointMass({ pt, r3, false, true });
		}
		_addPointMass({ { size.width / 2.0f, 0.0f }, 10.0f, false, true });
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
		auto subdt = dt / SUB_STEPS;
		_time = seconds.count();

		lock();
		for (int s = 0; s < SUB_STEPS; ++s) {
			for (auto& l : _links) l.constrain(_pointMasses.data(), subdt);
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
				p1.update(subdt);
			}
		}
		unlock();
	}
}