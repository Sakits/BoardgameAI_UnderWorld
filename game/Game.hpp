#ifndef __GAME_HPP
#define __GAME_HPP

#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class Game
{
public:
    const int n, win_n;
    int now_player;
    std::vector <std::vector<char>> board;
    std::vector <char> valids;

    const int feat_cnt = 3;
    std::vector <std::vector<char>> feat[3];

    Game(int _n, int _win_n);
    
    void init();

    int getActionSize();

    void get_next_state(int action);

    void get_valid_moves(int turn);

    double get_game_ended();

    void get_canonical_form();

    void get_feature();

    std::string string_representation();

    void display(py::array_t<char> pyboard);

/* -------------------------- py API --------------------------*/

    void get_board(py::array_t<char> pyboard, int player);

    py::array_t<char> return_board();

    py::array_t<char> return_feature();

    py::array_t<char> getInitBoard();

    py::tuple getBoardSize() const;

    py::tuple getFeatureSize() const;

    py::tuple getNextState(py::array_t<char> pyboard, int player, int action);

    py::array_t<char> getValidMoves(py::array_t<char> pyboard, int player, int turn);

    double getGameEnded(py::array_t<char> pyboard, int player);

    py::array_t<char> getCanonicalForm(py::array_t<char> pyboard, int player);

    py::array_t<char> getFeature(py::array_t<char> pyboard);

    std::string stringRepresentation(py::array_t<char> pyboard);
};


#endif