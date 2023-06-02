#pragma once

#include "types.h"

namespace buglife {

	class World {
	public:
		cv::Size2i size;

		World(const cv::Size2i& size) :
			size(size)
		{ }

	private:
		std::set<Object> _objects;
	};

}