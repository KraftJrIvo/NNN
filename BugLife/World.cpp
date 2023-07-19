#include "World.hpp"
#include "Food.h"

#define SUB_STEPS 4

namespace buglife {
	
	void World::generate() {
		for (int i = 0; i < 100; ++i) {
			float r1 = BL_RAND_FLOAT;
			float r2 = BL_RAND_FLOAT;
			float r3 = BL_RAND_FLOAT;
			add(_rocks, Rock(cv::Vec3b(20, 40, 60), { r1 * size.width, r2 * size.height }, std::max(r3, 0.1f), false));
		}
		_startTime = std::chrono::system_clock::now();
	}

	LookInfo World::look(Object* looker, const cv::Point2f& from, const cv::Point2f& to) {
		LookInfo li;
		float range = cv::norm(to - from);
		float minDist = INFINITY;
		for (auto& o : _objects) {
			if (o == looker)
				continue;
			float dist = cv::norm(o->pos - from);
			if (dist - o->radius < range) {
				cv::Point2f pt;
				bool intersects = intersectCircleBySegment(from, to, o->pos, o->radius, pt);
				if (intersects) {
					float dist2inter = cv::norm(pt - from);
					if (dist2inter < minDist) {
						minDist = dist2inter;
						li.seeSmth = true;
						li.dist = cv::norm(pt - from);
						li.color = o->color;
						li.target = o;
					}
				}
			}
		}
		return li;
	}

	void World::update() {		
		std::chrono::duration<double> seconds = std::chrono::system_clock::now() - _startTime;
		auto dt = seconds.count() - _time;
		_time = seconds.count();

		for (int s = 0; s < SUB_STEPS; ++s) {
			const float    response_coef = 0.75f;
			for (auto it1 = _objects.begin(); it1 != _objects.end(); ++it1) {
				for (auto it2 = _objects.begin(); it2 != _objects.end(); ++it2) {
					if (it1 == it2 || (*it1)->destroyed || (*it2)->destroyed || (!(*it1)->dynamic && !(*it2)->dynamic))
						continue;
					auto v = (*it1)->pos - (*it2)->pos;
					float dist2 = v.x * v.x + v.y * v.y;
					float min_dist = (*it1)->radius + (*it2)->radius;
					if (dist2 < min_dist * min_dist) {
						const float dist = sqrt(dist2);
						auto n = v / dist;
						const float mass_ratio_1 = (*it1)->radius / ((*it1)->radius + (*it2)->radius);
						const float mass_ratio_2 = (*it2)->radius / ((*it1)->radius + (*it2)->radius);
						const float delta = 0.5f * response_coef * (dist - min_dist);
						if ((*it1)->dynamic)
							(*it1)->pos -= n * (mass_ratio_2 * delta);
						if ((*it2)->dynamic)
							(*it2)->pos += n * (mass_ratio_1 * delta);
					}
				}
			}

			for (auto& o : _objects) {
				if (!o->destroyed) o->update(dt);
			}
		}

		for (auto& c : _creatures) {
			if (c.destroyed)
				continue;
			auto p1 = c.pos + c.radius * cv::Point2f(cos(c.orient), sin(c.orient));
			auto p2 = p1 + c.species.eyesight * cv::Point2f(cos(c.orient + c.eyeOrient), sin(c.orient + c.eyeOrient));
			LookInfo li = look(&c, p1, p2);
			c.live(li, dt);
			if (c.destroyed) {
				int ndrop = c.getNutrition() / (2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS);
				for (int i = 0; i < ndrop; ++i) {
					add(_foods, Food(c.pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f)));
				}
			} else {
				if (li.seeSmth && li.dist < BL_BITE_DIST && c.isBiting) {
					c.bite(*li.target);
					if (li.target->destroyed) {
						if (!li.target->isFood()) {
							int ndrop = li.target->getNutrition() / (2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS);
							for (int i = 0; i < ndrop; ++i) {
								add(_foods, Food(li.target->pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f)));
							}
						}
					}
				}
				if (c.isLaying) {
					add(_eggs, c.layEgg());
				}
			}
		}

		for (auto& o : _objects) {
			if (o->destroyed)
				continue;
			if (o->pos.x > size.width) {
				o->pos.x -= size.width;
				o->prvPos.x -= size.width;
			}
			if (o->pos.y > size.height) {
				o->pos.y -= size.height;
				o->prvPos.y -= size.height;
			}
			if (o->pos.x < 0) {
				o->pos.x += size.width;
				o->prvPos.x += size.width;
			}
			if (o->pos.y < 0) {
				o->pos.y += size.height;
				o->prvPos.y += size.height;
			}
		}

		for (auto& e : _eggs) {
			if (!e.destroyed && e.isHatching()) {
				add(_creatures, Creature(e));
				e.destroyed = true;
			}
		}

		if (_foods.size() < BL_MAX_FOODS) {
			add(_foods, Food({ BL_RAND_FLOAT * size.width, BL_RAND_FLOAT * size.height }));
		}

		checkClearPtr(_objects);
		checkClear(_rocks);
		checkClear(_foods);
		checkClear(_eggs);
		checkClear(_creatures);
	}
}