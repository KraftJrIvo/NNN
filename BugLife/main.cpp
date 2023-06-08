#include <iostream>
#include "..\NNN\NeuralNet.hpp"
#include "World.h"
#include "Drawer.h"

using namespace nnn;

int main()
{
	buglife::World w({ 100, 100 });

	w.generate();

	buglife::Drawer d(w, {1024, 1024});

	system("pause");
	//while (true) {
	//	std::this_thread::sleep_for(std::chrono::milliseconds(33));
	//}
}