#pragma once

#include <list>

#include "types.h"
#include "Creature.h"

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

		LookInfo look(Object* looker, const cv::Point2f& from, const cv::Point2f& to);

	private:
		std::chrono::system_clock::time_point _startTime;
		double _time;
		
		std::list<Rock> _rocks;
		std::list<Creature> _creatures;
		std::list<Food> _foods;
		std::list<Egg> _eggs;
		std::list<Object*> _objects;

		template<typename T>
		void add(std::list<T>& l, const T& o) {
			l.push_back(o);
			_objects.push_back(&l.back());
		}

		template<typename T>
		void checkClear(std::list<T>& l) {
			auto it = l.begin();
			while (it != l.end())
			{
				if (it->destroyed)
					l.erase(it++);
				else
					++it;
			}
		}

		template<typename T>
		void checkClearPtr(std::list<T*>& l) {
			auto it = l.begin();
			while (it != l.end())
			{
				if ((*it)->destroyed)
					l.erase(it++);
				else
					++it;
			}
		}
	};	
}