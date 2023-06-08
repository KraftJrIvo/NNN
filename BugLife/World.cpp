#include "World.h"

namespace buglife {
	
	void World::generate() {
		for (int i = 0; i < 100; ++i) {
			float r1 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
			float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
			float r3 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
			_objects.emplace_back(Object(cv::Vec3b(150, 150, 150), { r1 * size.width, r2 * size.height }, std::max(r3, 0.1f)));
		}
	}

	void World::update() {

	}
}