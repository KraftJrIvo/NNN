#pragma once

#include "types.h"

namespace buglife {

    void Object::update(double dt) {
        auto displacement = pos - prvPos;
        prvPos = pos;
        pos += displacement * 0.5;// +acceleration * (dt * dt);
        pos += vel * dt;
    }

    void Object::draw(cv::Mat img, const cv::Point2f& size, const cv::Point2f& offset, float scale, float coeff) const {
        cv::Point2f p = coeff * scale * pos + offset + (size / 2.0f);
        cv::circle(img, p, radius * coeff * scale, color, -1);
        auto lcolor = cv::Scalar(std::min(255, color[0] + 50), std::min(255, color[1] + 50), std::min(255, color[2] + 50));
        cv::circle(img, p, radius * coeff * scale, lcolor, 1);
    }

    float Species::getMaxEnergy() const {
        return 2.0f * 3.14159f * radius * radius;
    }

    void Species::mutateColor(cv::Vec3b& c) {
        float r = BL_RAND_FLOAT;
        if (r < 0.1) {
            c = { uchar(255.0f * BL_RAND_FLOAT), uchar(255.0f * BL_RAND_FLOAT), uchar(255.0f * BL_RAND_FLOAT) };
        }
        else {
            uchar red = uchar(255.0f * std::clamp((0.75f + BL_RAND_FLOAT / 2.0f) * float(color[0]) / 255.0f, 0.0f, 1.0f));
            uchar green = uchar(255.0f * std::clamp((0.75f + BL_RAND_FLOAT / 2.0f) * float(color[1]) / 255.0f, 0.0f, 1.0f));
            uchar blue = uchar(255.0f * std::clamp((0.75f + BL_RAND_FLOAT / 2.0f) * float(color[2]) / 255.0f, 0.0f, 1.0f));
            c = { blue, green, red };
        }
    }

    void Species::mutate(float crazyMutProb) {
        float r = BL_RAND_FLOAT;
        bool crazyMut = r < crazyMutProb;
        if (crazyMut || r < 0.1)
            radius *= 0.75f + BL_RAND_FLOAT * 0.5f;
        if (crazyMut || (r > 0.1f && r < 0.3f))
            mutateColor(color);
        if (crazyMut || (r > 0.3f && r < 0.4f))
            mutateColor(eggColor);
        if (crazyMut || (r > 0.4f && r < 0.5f))
            eyesight = std::clamp((0.75f + BL_RAND_FLOAT / 2.0f) * eyesight, 0.5f, BL_MAX_EYESIGHT);
        if (crazyMut || (r > 0.5f)) {
            brain.mutate(BL_BRAIN_W_MUT_PROB, crazyMut ? 1.0f : crazyMutProb);
        }
    }

    bool intersectCircleBySegment(const cv::Point2f& s, const cv::Point2f& e, const cv::Point2f& o, float r, cv::Point2f& point) {
        auto x0 = o.x;
        auto y0 = o.y;
        auto x1 = s.x;
        auto y1 = s.y;
        auto x2 = e.x;
        auto y2 = e.y;
        auto A = y2 - y1;
        auto B = x1 - x2;
        auto C = x2 * y1 - x1 * y2;
        auto a = A * A + B * B;
        float b, c;
        bool bnz = true;
        if (abs(B) >= 1e-8) {
            b = 2 * (A * C + A * B * y0 - B * B * x0);
            c = C * C + 2 * B * C * y0 - B * B * (r * r - x0 * x0 - y0 * y0);
        }
        else {
            b = 2 * (B * C + A * B * x0 - A * A * y0);
            c = C * C + 2 * A * C * x0 - A * A * (r * r - x0 * x0 - y0 * y0);
            bnz = false;
        }
        auto d = b * b - 4 * a * c; // discriminant
        if (d < 0) {
            return false;
        }

        // checks whether a point is within a segment
        auto within = [x1, y1, x2, y2](float x, float y) {
            auto d1 = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));  // distance between end-points
            auto d2 = sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));    // distance from point to one end
            auto d3 = sqrt((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));    // distance from point to other end
            auto delta = d1 - d2 - d3;
            return abs(delta) < 1e-8;                    // true if delta is less than a small tolerance
        };

        auto fx = [A, B, C](float x) {
            return -(A * x + C) / B;
        };

        auto fy = [A, B, C](float y) {
            return -(B * y + C) / A;
        };

        auto rxy = [&point, within](float x, float y) {
            if (within(x, y))
                point = { x, y };
        };

        float x, y;
        if (d == 0.0) {
            // line is tangent to circle, so just one intersect at most
            if (bnz) {
                x = -b / (2 * a);
                y = fx(x);
                rxy(x, y);
            }
            else {
                y = -b / (2 * a);
                x = fy(y);
                rxy(x, y);
            }
        }
        else {
            // two intersects at most
            d = sqrt(d);
            if (bnz) {
                x = (-b + d) / (2 * a);
                y = fx(x);
                rxy(x, y);
                x = (-b - d) / (2 * a);
                y = fx(x);
                rxy(x, y);
            }
            else {
                y = (-b + d) / (2 * a);
                x = fy(y);
                rxy(x, y);
                y = (-b - d) / (2 * a);
                x = fy(y);
                rxy(x, y);
            }
        }

        return cv::norm(point - s) < cv::norm(e - s);
    }

}