#include "Drawer.h"

namespace life2d
{
	Drawer::Drawer(World& w, cv::Size size, float scale, std::string windowName) :
		_w(w),
		_size(size),
		_windowName(windowName.length() ? windowName : "buglife"),
		_zoom(1.0f),
		_scale(0.01f),
		_offset(_scale * size.width / 2.0, _scale * size.height / 2.0),
		_coeff(1.0f),
		_bounds({ {-1.5f, -1.5f, -1.5f }, {1.5f, 1.5f, 1.5f} }),
		_mouse(0, 0),
		_lclick(false)
	{
		_img.create(size, CV_8UC3);

		_redrawer = std::thread([&]() {
			while (true) {
				_draw();
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			});

		_bounds = { {-1.5f, -1.5f, -1.5f }, {1.5f, 1.5f, 1.5f} };

	}
	//---------------------------------------------------------------------------------------------------------------
	void Drawer::_updateBounds(const cv::Point3f& pos) {
		_bounds.first.x = std::min(_bounds.first.x, pos.x);
		_bounds.first.y = std::min(_bounds.first.y, pos.y);
		_bounds.first.z = std::min(_bounds.first.z, pos.z);
		_bounds.second.x = std::max(_bounds.second.x, pos.x);
		_bounds.second.y = std::max(_bounds.second.y, pos.y);
		_bounds.second.z = std::max(_bounds.second.z, pos.z);
	}
	//---------------------------------------------------------------------------------------------------------------
	void Drawer::_changeZoom(float val) {
		_zoom += val;
	}
	//---------------------------------------------------------------------------------------------------------------
	cv::Point2f Drawer::_getScaledProj(const cv::Point2f& pt) {
		return _coeff * _getPtProj(_scale * pt) + _offset + (_size / 2.0f);
	}
	//---------------------------------------------------------------------------------------------------------------
	cv::Point2f Drawer::_getPtProj(const cv::Point2f& pt) {
		return { pt.x, pt.y };
	}
	//---------------------------------------------------------------------------------------------------------------
	cv::Point2f Drawer::_getPos2d(const cv::Point2f& pt) {
		auto nspt = (pt - _size / 2.0f - _offset) / _coeff;
		return { nspt.x / _scale, nspt.y / _scale };
	}
	cv::Point2f closestPointOnLine2(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p) {
		cv::Point2f line = p2 - p1;
		float line_length = std::sqrt(line.x * line.x + line.y * line.y);
		cv::Point2f line_direction = line / line_length;
		cv::Point2f p1_to_p = p - p1;
		float projection = p1_to_p.dot(line_direction);
		cv::Point2f closest_point = p1 + line_direction * projection;
		return closest_point;
	}
	//---------------------------------------------------------------------------------------------------------------
	void Drawer::_draw()
	{
		_img.setTo(cv::Scalar(0, 0, 0));

		float maxRealSide = std::max(_bounds.second.x - _bounds.first.x, std::max(_bounds.second.y - _bounds.first.y, _bounds.second.z - _bounds.first.z));
		float minVisSide = std::min(_size.x, _size.y);
		_pixPerMeter = minVisSide / 3.0;
		_coeff = _zoom * _pixPerMeter;

		cv::Point2f p1 = _getScaledProj({ 0, 0 });
		cv::Point2f p2 = _getScaledProj({ (float)_w.size.width, 0 });
		cv::Point2f p3 = _getScaledProj({ (float)_w.size.width, (float)_w.size.height });
		cv::Point2f p4 = _getScaledProj({ 0, (float)_w.size.height });
		cv::line(_img, p1, p2, cv::Scalar(100, 100, 100), 2);
		cv::line(_img, p2, p3, cv::Scalar(100, 100, 100), 2);
		cv::line(_img, p3, p4, cv::Scalar(100, 100, 100), 2);
		cv::line(_img, p4, p1, cv::Scalar(100, 100, 100), 2);

		_w.lock();
		auto c2d = _getPos2d(_mouse);
		_w._pointMasses[0].pos = c2d;
		for (auto& p : _w._planes)
			p.draw(_img, _size, _offset, _scale, _coeff);
		for (auto& pm : _w._pointMasses)
			pm.draw(_img, _size, _offset, _scale, _coeff);
		for (auto& l : _w._links)
			l.draw(_img, _size, _offset, _scale, _coeff, _w._pointMasses.data());
		for (auto& p : _w._polygons)
			p.draw(_img, _size, _offset, _scale, _coeff, _w._pointMasses.data());
		_w.unlock();

		auto onMouse = [](int event, int x, int y, int flags, void* _data) {
			static bool lmb = false;

			((Drawer*)_data)->_mouse.x = x;
			((Drawer*)_data)->_mouse.y = y;

			static cv::Point2f lastPos = { (float)x, (float)y };
			static cv::Point2f lastOffset = ((Drawer*)_data)->_offset;

			if (event == cv::EVENT_LBUTTONDOWN && !lmb) {
				lastPos = { (float)x, (float)y };
				lastOffset = ((Drawer*)_data)->_offset;
			}

			static bool rmb = false;
			if (event == cv::EVENT_RBUTTONDOWN) rmb = true;
			if (event == cv::EVENT_RBUTTONUP && rmb) {
				rmb = false;
			}

			if (event == cv::EVENT_LBUTTONUP && lmb) {
				auto c2d = ((Drawer*)_data)->_getPos2d(((Drawer*)_data)->_mouse);
				// click
			}

			if (event == cv::EVENT_LBUTTONDOWN) lmb = true;
			if (event == cv::EVENT_LBUTTONUP) {
				lmb = false;

			}
			((Drawer*)_data)->_lclick = (event == cv::EVENT_LBUTTONUP);

			if (lmb) {
				((Drawer*)_data)->_offset = lastOffset - (lastPos - cv::Point2f(x, y));
			}

			if (event == cv::EVENT_MOUSEWHEEL) {
				float mwdelta = cv::getMouseWheelDelta(flags) / 500.0;
				float prvZoom = ((Drawer*)_data)->_zoom;
				auto c2d = ((Drawer*)_data)->_getPos2d(((Drawer*)_data)->_mouse);
				((Drawer*)_data)->_zoom = mwdelta > 0 ?
					(((Drawer*)_data)->_zoom * (1.0 + mwdelta)) :
					(((Drawer*)_data)->_zoom / (1.0 - mwdelta));
				float maxRealSide = std::max(((Drawer*)_data)->_bounds.second.x - ((Drawer*)_data)->_bounds.first.x,
					std::max(((Drawer*)_data)->_bounds.second.y - ((Drawer*)_data)->_bounds.first.y,
						((Drawer*)_data)->_bounds.second.z - ((Drawer*)_data)->_bounds.first.z));
				float minVisSide = std::min(((Drawer*)_data)->_size.x, ((Drawer*)_data)->_size.y);
				((Drawer*)_data)->_pixPerMeter = minVisSide / 3.0;
				((Drawer*)_data)->_coeff = ((Drawer*)_data)->_zoom * ((Drawer*)_data)->_pixPerMeter;
				float sgn = mwdelta / abs(mwdelta);
				((Drawer*)_data)->_offset -= ((Drawer*)_data)->_getScaledProj(c2d) - ((Drawer*)_data)->_getScaledProj(((Drawer*)_data)->_getPos2d(((Drawer*)_data)->_mouse));
			}
		};
		cv::imshow(_windowName, _img);
		cv::setMouseCallback(_windowName, onMouse, this);
		cv::waitKey(1);
	}
}