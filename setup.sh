g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` game/Game.hpp game/Game.cpp mcts/MCTS.cpp -o libcpp`python3-config --extension-suffix`