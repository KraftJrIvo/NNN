#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "..\NNN\Drawer.hpp"
#include "World.h"
#include "Drawer.h"

int main()
{
	life2d::World w({ 100, 100 });
	w.generate();

	life2d::Drawer d(w, { 1024, 1024 });

	while (true) {
		w.update();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}