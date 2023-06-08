#pragma once

#include "types.h"

namespace buglife {

	class Drawer;

	class World {
	friend class Drawer;
	public:
		cv::Size2i size;

		World(const cv::Size2i& size) :
			size(size)
		{ }

		void generate();
		void update();

	private:
		double _startTime = 0;
		std::vector<Object> _objects;
	};

}