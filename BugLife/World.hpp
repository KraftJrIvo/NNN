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
			size(size),
			_temperature(cv::Mat(cv::Size2i(int(size.width * BL_TEMP_RES), int(size.height * BL_TEMP_RES)), CV_32FC1, 0.0f)),
			_nextTemperature(cv::Mat(cv::Size2i(int(size.width* BL_TEMP_RES), int(size.height* BL_TEMP_RES)), CV_32FC1, 0.0f))
		{ }

		void generate();
		void update();

		LookInfo look(Object* looker, const cv::Point2f& from, const cv::Point2f& to);

		void lock() {
			_lock.lock();
		}
		void unlock() {
			_lock.unlock();
		}

		void timescale() {
			_timescale = (_timescale > 1.0f) ? 1.0f : BL_TIMESCALE;
		}

		void save(std::ofstream& out) {
			size_t sz = _rocks.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& r : _rocks) r.save(out); 
			sz = _trees.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& t : _trees) t.save(out);
			sz = _creatures.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& c : _creatures) c.save(out);
			sz = _foods.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& f : _foods) f.save(out);
			sz = _eggs.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& e : _eggs) e.save(out);
			sz = _poison.size(); out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
			for (auto& p : _poison) p.save(out);
			out.write(reinterpret_cast<const char*>(&_accTime), sizeof(float));
		}

		void save(const std::string& filePath) {
			std::ofstream out(filePath, std::ios::binary);
			save(out);
		}

		void load(std::ifstream& in) {
			size_t sz;
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _rocks.resize(sz);
			for (auto& r : _rocks) { r.load(in); _objects.push_back(&r); }
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _trees.resize(sz);
			for (auto& t : _trees) { t.load(in); _objects.push_back(&t); }
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _creatures.resize(sz);
			for (auto& c : _creatures) { c.load(in); _objects.push_back(&c); }
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _foods.resize(sz);
			for (auto& f : _foods) { f.load(in); _objects.push_back(&f); }
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _eggs.resize(sz);
			for (auto& e : _eggs) { e.load(in); _objects.push_back(&e); }
			in.read(reinterpret_cast<char*>(&sz), sizeof(size_t)); _poison.resize(sz);
			for (auto& p : _poison) { p.load(in); _objects.push_back(&p); }
			in.read(reinterpret_cast<char*>(&_accTime), sizeof(float));
		}

		void load(const std::string& filePath) {
			std::ifstream in(filePath, std::ios::binary); 
			load(in);
		}

	private:
		std::chrono::system_clock::time_point _startTime;
		double _time, _lastSaveTime, _timescale = 1.0;
		float _accTime = 0.0f;

		cv::Mat _temperature, _nextTemperature;
		
		bool _first = true;

		std::list<Rock> _rocks;
		std::list<Tree> _trees;
		std::list<Creature> _creatures;
		std::list<Food> _foods;
		std::list<Egg> _eggs;
		std::list<Poison> _poison;
		std::list<Object*> _objects;

		std::mutex _lock;

		void _updateTemperature(float dt);

		template<typename T>
		void _add(std::list<T>& l, const T& o) {
			l.push_back(o);
			_objects.push_back(&l.back());
		}

		template<typename T>
		void _checkClear(std::list<T>& l) {
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
		void _checkClearPtr(std::list<T*>& l) {
			auto it = l.begin();
			while (it != l.end())
			{
				if ((*it)->destroyed) {
					for (auto& c : _creatures) {
						if (c.destroyed)
							continue;
						if (c.target == (*it)) {
							c.target = nullptr;
						}
					}
					l.erase(it++);
				} else if (it != l.end())
					++it;
			}
		}
	};	
}