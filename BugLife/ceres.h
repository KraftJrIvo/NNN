#pragma once

//class PushOutFunctor {
	//public:
	//	explicit PushOutFunctor(float r1, float r2) : _d(r1 + r2)
	//	{}
	//
	//	static ceres::CostFunction* Create(float r1, float r2) {
	//		return (new ceres::AutoDiffCostFunction<
	//			PushOutFunctor, 1, 2, 2>(
	//				new PushOutFunctor(r1, r2)));
	//	}
	//
	//	template <typename T>
	//	bool operator()(const T* const pos1, const T* const pos2, T* residuals) const {
	//
	//		Eigen::Matrix<T, 2, 1> p1(pos1);
	//		Eigen::Matrix<T, 2, 1> p2(pos2);
	//
	//		auto a = (p1 - p2).norm() - (T)_d;
	//		residuals[0] = (a < (T)0) ? a : (T)0;
	//
	//		return true;
	//	}
	//
	//private:
	//	float _d;
	//};
	//
	//class PushOutFunctor2 : public ceres::SizedCostFunction<1, 2, 2>
	//{
	//public:
	//	PushOutFunctor2(float r1, float r2) : _d(r1 + r2)
	//	{ }
	//
	//	virtual bool Evaluate(
	//		double const* const* parameters,
	//		double* residuals,
	//		double** jacobians) const override {
	//
	//		Eigen::Matrix<double, 2, 1> p1(parameters[0]);
	//		Eigen::Matrix<double, 2, 1> p2(parameters[1]);
	//
	//		auto a = (p1 - p2).norm() - _d;
	//		bool push = a < 0;
	//		residuals[0] = push ? a : 0;
	//
	//
	//		if (jacobians) {
	//			auto dex = (p1.x() - p2.x()) / sqrt(std::max(abs(p1.x() - p2.x()), 0.001));
	//			auto dey = (p1.y() - p2.y()) / sqrt(std::max(abs(p1.y() - p2.y()), 0.001));
	//			if (jacobians[0]) {
	//				Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> j0(jacobians[0]);
	//				j0.setZero();
	//				if (push) {
	//					jacobians[0][0] = dex;
	//					jacobians[0][1] = dey;
	//				} 
	//			}
	//			if (jacobians[1]) {
	//				Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> j1(jacobians[1]);
	//				j1.setZero();
	//				if (push) {
	//					jacobians[1][0] = -dex;
	//					jacobians[1][1] = -dey;
	//				}
	//			}
	//		}
	//
	//		return true;
	//	}
	//
	//private:
	//	float _d;
	//};
		//ceres::Problem problem;

		//for (auto& o : _objects) {
		//	problem.AddParameterBlock(o.pos.data(), 2);
		//	if (!o.dynamic)
		//		problem.SetParameterBlockConstant(o.pos.data());
		//}

		//for (int i = 0; i < _objects.size(); ++i) {
		//	for (int j = i + 1; j < _objects.size(); ++j) {
		//		auto dist = (_objects[i].pos - _objects[j].pos).norm();
		//		if (dist < 3 * (_objects[i].radius + _objects[j].radius)) {
		//			//auto cost = PushOutFunctor::Create(_objects[i].radius, _objects[j].radius);
		//			auto cost = new PushOutFunctor2(_objects[i].radius, _objects[j].radius);
		//			problem.AddResidualBlock(cost, nullptr, _objects[i].pos.data(), _objects[j].pos.data());
		//		}
		//	}
		//}

		//ceres::Solver::Options options;
		//options.max_num_consecutive_invalid_steps = 10;
		//options.max_consecutive_nonmonotonic_steps = 10;
		//options.function_tolerance = 0;
		//options.gradient_tolerance = 0.000001;
		//options.parameter_tolerance = 0;
		//options.max_num_iterations = 5;
		//options.max_linear_solver_iterations = 5;
		////options.minimizer_progress_to_stdout = true;
		//options.linear_solver_type = ceres::DENSE_QR;
		//options.num_threads = 1;

		//ceres::Solver::Summary summary;
		//ceres::Solve(options, &problem, &summary);