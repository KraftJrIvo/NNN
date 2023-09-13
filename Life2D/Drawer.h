#pragma once

#include "World.h"

namespace life2d
{

	class Drawer
	{
	public:
		Drawer(World& w, cv::Size size = { 512, 512 }, float scale = 1.0f, std::string windowName = "");

	private:
		std::thread _redrawer;

		World& _w;

		std::string _windowName;
		float _zoom;
		cv::Point2f _mouse;
		bool _lclick;
		float _scale;
		float _coeff;
		cv::Point2f _offset;
		cv::Point2f _size;
		float _pixPerMeter;

		cv::Mat _img;
		std::pair<cv::Point3f, cv::Point3f> _bounds;

		void _changeZoom(float val);
		void _updateBounds(const cv::Point3f& pos);

		cv::Point2f _getPtProj(const cv::Point2f& pt);
		cv::Point2f  _getPos2d(const cv::Point2f& pt);
		cv::Point2f _getScaledProj(const cv::Point2f& pt);

		void _draw();
	};
}