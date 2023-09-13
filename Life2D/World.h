#pragma once

#include <mutex>
#include <fstream>
#include <list>

#include "types.h"

namespace life2d {

	class Drawer;

	class World {
		friend class Drawer;
	public:
		cv::Size2i size;

		World(const cv::Size2i& size) :
			size(size)
		{ 
			_startTime = std::chrono::system_clock::now();
		}

		void generate();
		void update();

		void lock() {
			_lock.lock();
		}
		void unlock() {
			_lock.unlock();
		}

		void save(std::ofstream& out) {
		}
		void save(const std::string& filePath) {
			std::ofstream out(filePath, std::ios::binary);
			save(out);
		}

		void load(std::ifstream& in) {
		}
		void load(const std::string& filePath) {
			std::ifstream in(filePath, std::ios::binary);
			load(in);
		}

	private:
		std::mutex _lock;

		std::chrono::system_clock::time_point _startTime;
		double _time = 0.0;
		bool _first = true;

		std::vector<Plane> _planes;
		std::vector<PointMass> _pointMasses;
		std::vector<PointMassLink> _links;
		std::vector<Polygon> _polygons;

		size_t _addPlane(const Plane& p);
		size_t _addPointMass(const PointMass& pm);
		size_t _addLink(const PointMassLink& l);
		size_t _addPolygon(const Polygon& p);
	};
}