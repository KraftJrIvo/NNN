#include "World.hpp"
#include "Food.h"

#define SUB_STEPS 4

namespace buglife {
	
	void World::generate() {
		for (int i = 0; i < 100; ++i) {
			float r1 = BL_RAND_FLOAT;
			float r2 = BL_RAND_FLOAT;
			float r3 = BL_RAND_FLOAT;
			_add(_rocks, Rock(cv::Vec3b(40, 40, 40), { r1 * size.width, r2 * size.height }, std::max(r3, 0.1f), false));
			//_add(_rocks, Rock(cv::Vec3b(20, 40, 60), { r1 * size.width, r2 * size.height }, std::max(r3, 0.1f), false));
		}
		for (int i = 0; i < 100; ++i) {
			float r1 = BL_RAND_FLOAT;
			float r2 = BL_RAND_FLOAT;
			float r3 = BL_RAND_FLOAT;
			_add(_trees, Rock(cv::Vec3b(20, 40, 60), { r1 * size.width, r2 * size.height }, std::max(r3, 0.1f), false));
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
			auto opos = o->pos - from;
			if (opos.x < -size.width/2) opos.x += size.width;
			if (opos.x > size.width/2) opos.x -= size.width;
			if (opos.y < -size.height / 2) opos.y += size.height;
			if (opos.y > size.height / 2) opos.y -= size.height;
			float dist = cv::norm(opos);
			if (dist - o->radius < range) {
				cv::Point2f pt;
				bool intersects = intersectCircleBySegment(from, to, from + opos, o->radius, pt);
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
		auto dt = _timescale * (_first ? 0 : (seconds.count() - _time));
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
				if (!o->destroyed) o->update(dt / (float)SUB_STEPS);
			}
		}

		for (auto& c : _creatures) {
			//c.species.save("oof.spc");
			if (c.destroyed)
				continue;
			auto p1 = c.pos + c.radius * cv::Point2f(cos(c.orient), sin(c.orient));
			auto p2 = p1 + c.species.eyesight * cv::Point2f(cos(c.orient + c.eyeOrient), sin(c.orient + c.eyeOrient));
			LookInfo li = look(&c, p1, p2);
			c.live(li, dt, size);
			if (c.destroyed) {
				int ndrop = c.getNutrition() / (2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS);
				for (int i = 0; i < ndrop; ++i) {
					_add(_foods, Food(c.pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f)));
				}
			} else {
				//if (c.isBiting) std::cout << li.dist << std::endl;
				if (li.seeSmth && li.dist < BL_BITE_DIST && c.isBiting) {
					c.bite(*li.target);
					if (li.target->destroyed) {
						if (!li.target->isFood()) {
							int ndrop = li.target->getNutrition() / (2.0f * 3.14159f * BL_FOOD_RADIUS * BL_FOOD_RADIUS);
							for (int i = 0; i < ndrop; ++i) {
								_add(_foods, Food(li.target->pos + cv::Point2f(BL_RAND_FLOAT / 100.0f, BL_RAND_FLOAT / 100.0f)));
							}
						}
					}
				}
				if (c.isLaying && _creatures.size() < BL_MAX_CREATURES) {
					if (c.tryLay()) {
						lock();
						_add(_eggs, c.layEgg(0.0f));
						_add(_eggs, c.layEgg(1.0f));
						unlock();
					}
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

		lock();
		for (auto& e : _eggs) {
			if (!e.destroyed && e.isHatching()) {
				_add(_creatures, Creature(e));
				e.destroyed = true;
			}
		}

		while (_foods.size() < BL_MAX_FOODS) {
			int r = BL_RAND_FLOAT * (float)(_trees.size() - 1);
			auto it = _trees.begin(); std::advance(it, r);
			float& temp = _temperature.at<float>(int(it->pos.y * BL_TEMP_RES), int(it->pos.x * BL_TEMP_RES));
			if (BL_RAND_FLOAT < temp || BL_RAND_FLOAT < 0.001)
				_add(_foods, Food(it->pos + cv::Point2f(BL_RAND_FLOAT * 0.01f - 0.005f, BL_RAND_FLOAT * 0.01f - 0.005f)));
		}
		/*while (_foods.size() < BL_MAX_FOODS) {
			auto pos = cv::Point2f({ BL_RAND_FLOAT * size.width, BL_RAND_FLOAT * size.height });
			float& temp = _temperature.at<float>(int(pos.y * BL_TEMP_RES), int(pos.x * BL_TEMP_RES));
			if (BL_RAND_FLOAT < temp || BL_RAND_FLOAT < 0.001)
				_add(_foods, Food(pos));
		}*/

		/*while (_poison.size() < BL_MAX_POISON) {
			_add(_poison, Poison({ BL_RAND_FLOAT * size.width, BL_RAND_FLOAT * size.height }));
		}*/
		while (_poison.size() < BL_MAX_POISON) {
			int r = BL_RAND_FLOAT * (float)(_trees.size() - 1);
			auto it = _trees.begin(); std::advance(it, r);
			_add(_poison, Poison(it->pos + cv::Point2f(BL_RAND_FLOAT * 0.01f - 0.005f, BL_RAND_FLOAT * 0.01f - 0.005f)));
		}
		for (auto& p : _poison) {
			if (!p.destroyed && p.pos.x < 2 || p.pos.x > size.width - 2 || p.pos.y < 2 || p.pos.y > size.height - 2) {
				p.destroyed = true;
			}
		}
		
		_checkClearPtr(_objects);
		_checkClear(_rocks);
		_checkClear(_foods);
		_checkClear(_poison);
		_checkClear(_eggs);
		_checkClear(_creatures);
		unlock();

		_updateTemperature(dt);

		if (_accTime - _lastSaveTime > BL_AUTOSAVE_INTERVAL && _creatures.size() > 30) {
			save("world.w");
			_lastSaveTime = _accTime;
		}


		_accTime += dt;
		_first = false;

		if (_accTime > 1000.0f && _creatures.size() == 0) {
			load("world.w");
		}
	}

	void World::_updateTemperature(float dt) {
		for (auto& c : _creatures) {
			if (c.destroyed)
				continue;
			float& temp = _temperature.at<float>(int(c.pos.y * BL_TEMP_RES), int(c.pos.x * BL_TEMP_RES));
			int fwdx = (c.pos.y + 0.5 * sin(c.orient + c.eyeOrient)) * BL_TEMP_RES;
			int fwdy = (c.pos.x + 0.5 * cos(c.orient + c.eyeOrient)) * BL_TEMP_RES;
			float& tempFwd = _temperature.at<float>((fwdx >= 0) ? ((fwdx < size.width) ? fwdx : fwdx - size.width) : fwdx + size.width, (fwdy >= 0) ? ((fwdy < size.height) ? fwdy : fwdy - size.height) : fwdy + size.height);
			c.handleTemperature(temp, tempFwd);
			temp = BL_CREATURE_TEMP;// += (BL_CREATURE_TEMP - temp) / 2;
		}
		for (int i = 0; i < _temperature.rows; ++i)
			for (int j = 0; j < _temperature.cols; ++j) {
				float airdtemp = dt / _timescale * (BL_NIGHT_D_TEMP + (BL_DAY_D_TEMP - BL_NIGHT_D_TEMP) * (1.0f + (float)sin((2.0f * 3.14f * ((float)(j / BL_TEMP_RES)/(float)size.width)) + _accTime / BL_DAY_LEN)) / 2.0f);
				_nextTemperature.at<float>(i, j) = std::clamp(_temperature.at<float>(i, j) + BL_TEMP_DIFFUSIVITY / (float)sqrt(sqrt(_timescale)) * (_temperature.at<float>((i > 0) ? (i - 1) : (int(BL_TEMP_RES * size.height) - 1), j) + _temperature.at<float>((i < int(BL_TEMP_RES * size.height) - 1) ? (i + 1) : 0, j) + _temperature.at<float>(i, (j > 0) ? (j - 1) : (int(BL_TEMP_RES * size.width) - 1)) + _temperature.at<float>(i, (j < int(BL_TEMP_RES * size.width) - 1) ? (j + 1) : 0) - 4.0f * _temperature.at<float>(i, j)) + airdtemp, 0.0f, 1000.0f);
			}
		cv::Mat tmp = _temperature;
		_temperature = _nextTemperature;
		_nextTemperature = tmp;
	}
}