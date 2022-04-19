#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "../game/Game.hpp"

const double eps = 1e-8;
using uint = unsigned int;

class MCTS
{
public:
    class node
    {
        friend class MCTS;

        int Ns;
        double Es;
        std::vector <int> son, Nsa;
        std::vector <double> Qsa, Ps;
        Game game;

        node(const Game &_game):game(_game) 
        {
            Ns = 0;
            game.get_valid_moves();
            Es = game.get_game_ended();
            Ps.clear();

            son.resize(game.getActionSize());
            Nsa.resize(game.getActionSize());
            Qsa.resize(game.getActionSize());
            for (uint i = 0; i < son.size(); i++)
                Qsa[i] = Nsa[i] = son[i] = 0;
        }
    };


    Game game;
    const py::object nn;
    const int num_MCTS_sims;
    const double cpuct;
    bool status;
    int root;
    double _v;
    std::vector <node> mcts;
    std::vector < std::pair<int, int> > path;
    std::vector <double> probs;
    std::vector <int> counts;

    MCTS(const Game &_game, const py::object _nn, int _num_MCTS_sims, double _cpuct) 
        :game(_game), nn(_nn), num_MCTS_sims(_num_MCTS_sims), cpuct(_cpuct) 
    {
        reset();
    }

    void reset()
    {
        _v = root = status = 0;
        mcts.clear();
        path.clear();
        game.init();
        probs.resize(game.getActionSize());
        counts.resize(game.getActionSize());
        mcts.push_back(node(game));
    }

    py::array_t<double> return_probs(int temp, bool arena = false)
    {
        int siz = game.getActionSize();
        int mx = 0, mxpos = 0;
        for (int i = 0; i < siz; i++)
            if (counts[i] > mx)
            {
                mx = counts[i];
                mxpos = i;
            }

        if (!temp)
        {
            for (int i = 0; i < siz; i++)
                probs[i] = 0;
            probs[mxpos] = 1;
        }
        else
        {
            int sum = 0;
            for (int i = 0; i < siz; i++)
                sum += counts[i];
            assert(sum > 0);
            for (int i = 0; i < siz; i++)
                probs[i] = 1.0 * counts[i] / sum;
        }

        if (arena)
        {
            printf("win_rate : %.3lf\n", (mcts[root].Qsa[mxpos] + 1) / 2);
            root = mcts[root].son[mxpos];
        }

        auto pyprobs = py::array_t<double>(probs.size());
        double* ptr = static_cast<double *>(pyprobs.request().ptr);

        for (uint i = 0; i < probs.size(); i++)
            ptr[i] = probs[i];

        return pyprobs;
    }

    void find_root(py::array_t<char> canonicalBoard)
    {
        bool flag = 0;
        std::string s = game.stringRepresentation(canonicalBoard);

        if (s == mcts[root].game.string_representation())
            return;

        for (uint i = 0; i < mcts[root].son.size(); i++)
            if (mcts[root].son[i])
            {
                if (s == mcts[mcts[root].son[i]].game.string_representation())
                {
                    root = mcts[root].son[i];
                    flag = 1;
                    break;
                }
            }

        if (!flag)
        {
            game.get_board(canonicalBoard, 1);
            mcts.push_back(node(game));
            root = mcts.size() - 1;
        }
    }

    py::array_t<double> getActionProb(py::array_t<char> canonicalBoard, int temp = 1)
    {
        find_root(canonicalBoard);
        int x = root;

        for (int i = 0; i < num_MCTS_sims; i++)
            search(x);

        int siz = mcts[x].game.getActionSize();
        for (int i = 0; i < siz; i++)
            counts[i] = mcts[x].son[i] ? mcts[x].Nsa[i] : 0;

        return return_probs(temp, true);
    }

    py::array_t<double> getExpertProb(py::array_t<char> canonicalBoard, int temp = 1, bool prune = false)
    {
        find_root(canonicalBoard);
        int x = root;

        int siz = mcts[x].game.getActionSize();
        for (int i = 0; i < siz; i++)
            counts[i] = mcts[x].son[i] ? mcts[x].Nsa[i] : 0;

        if (prune)
        {
            int mx = 0, mxpos = 0;
            for (int i = 0; i < siz; i++)
                if (counts[i] > mx)
                {
                    mx = counts[i];
                    mxpos = i;
                }

            double mxv = mcts[x].Qsa[mxpos] + cpuct * mcts[x].Ps[mxpos] * sqrt(mcts[x].Ns) / (counts[mxpos] + 1);

            for (int i = 0; i < siz; i++)
            {
                if (i == mxpos || counts[i] <= 0)
                    continue;

                int desired = (int)ceil(sqrt(2 * mcts[x].Ps[i] * mcts[x].Ns));
                double v_const = mcts[x].Qsa[i] + cpuct * mcts[x].Ps[i] * sqrt(mcts[x].Ns);
                for (int j = 0; j < desired; j++)
                {
                    if (counts[i] <= 0)
                        break;
                    if (v_const / counts[i] < mxv)
                        counts[i] --;
                }
            }
        }

        return return_probs(temp);
    }

    void processResult(py::array_t<double> pi, double value)
    {
        if (!status)
        {
            int x = path.back().first;
            path.pop_back();

            double sum = 0;
            double* ptr = static_cast<double *>(pi.request().ptr);
            for (uint i = 0; i < mcts[x].Ps.size(); i++)
                mcts[x].Ps[i] = ptr[i] * mcts[x].game.valids[i], sum += mcts[x].Ps[i];
            
            if (sum > eps)
            {
                for (uint i = 0; i < mcts[x].Ps.size(); i++)
                    mcts[x].Ps[i] /= sum;
            }
            else
            {
                puts("All valid moves were masked, do workaround");
                exit(1);
            }

            mcts[x].Ns = 0;
            _v = -value;
        }

        reverse(path.begin(), path.end());
        for (uint j = 0; j < path.size(); j++)
        {
            int x = path[j].first, i = path[j].second;
            mcts[x].Qsa[i] = (mcts[x].Nsa[i] * mcts[x].Qsa[i] + _v) / (mcts[x].Nsa[i] + 1);
            mcts[x].Nsa[i] ++;
            mcts[x].Ns ++;
            _v *= -1;
        }

        path.clear();
    }

    std::pair<bool, py::array_t<char>> findLeafToProcess(py::array_t<char> canonicalBoard)
    {
        find_root(canonicalBoard);
        return rollout(root);
    }

    std::pair<bool, py::array_t<char>> rollout(int x)
    {
        if (fabs(mcts[x].Es) > eps)
        {
            status = 1;
            _v = -mcts[x].Es;
            return std::make_pair(false, mcts[x].game.return_feature());
        }

        if (mcts[x].Ps.empty())
        {
            status = 0;
            mcts[x].Ps.resize(mcts[x].game.getActionSize());
            path.push_back(std::make_pair(x, -1));
            return std::make_pair(true, mcts[x].game.return_feature());
        }

        double cur_best = -1e9;
        int best_act = -1;

        int siz = mcts[x].game.getActionSize();
        for (int i = 0; i < siz; i++)
            if (mcts[x].game.valids[i])
            {
                if (x == root && mcts[x].son[i] && mcts[x].Nsa[i] < sqrt(2 * mcts[x].Ps[i] * mcts[x].Ns))
                {
                    best_act = i;
                    break;
                }

                double value = mcts[x].Qsa[i] + cpuct * mcts[x].Ps[i] * sqrt(mcts[x].Ns) / (1 + mcts[x].Nsa[i]);

                if (value > cur_best)
                {
                    cur_best = value;
                    best_act = i;
                }
            }

        if (!mcts[x].son[best_act])
        {
            Game next_state = mcts[x].game;
            next_state.get_next_state(best_act);
            next_state.get_canonical_form();

            mcts[x].son[best_act] = mcts.size();
            mcts.push_back(node(next_state));
        }
        path.push_back(std::make_pair(x, best_act));
        
        return rollout(mcts[x].son[best_act]);
    }

    double search(int x)
    {
        if (fabs(mcts[x].Es) > eps)
            return -mcts[x].Es;  

        if (mcts[x].Ps.empty())
        {
            auto nnet = nn(mcts[x].game.return_feature()).cast<std::pair<py::array_t<double>, double>>();
            mcts[x].Ps.resize(mcts[x].game.getActionSize());
            double sum = 0;
            double* ptr = static_cast<double *>(nnet.first.request().ptr);
            for (uint i = 0; i < mcts[x].Ps.size(); i++)
                mcts[x].Ps[i] = ptr[i] * mcts[x].game.valids[i], sum += mcts[x].Ps[i];

            if (sum > eps)
            {
                for (uint i = 0; i < mcts[x].Ps.size(); i++)
                    mcts[x].Ps[i] /= sum;
            }
            else
            {
                puts("All valid moves were masked, do workaround");
                exit(1);
            }

            return -nnet.second;
        }

        double cur_best = -1e9;
        int best_act = -1;

        int siz = mcts[x].game.getActionSize();
        for (int i = 0; i < siz; i++)
            if (mcts[x].game.valids[i])
            {
                double value = mcts[x].Qsa[i] + cpuct * mcts[x].Ps[i] * sqrt(mcts[x].Ns) / (1 + mcts[x].Nsa[i]);

                if (value > cur_best)
                {
                    cur_best = value;
                    best_act = i;
                }
            }

        if (!mcts[x].son[best_act])
        {
            Game next_state = mcts[x].game;
            next_state.get_next_state(best_act);
            next_state.get_canonical_form();

            mcts[x].son[best_act] = mcts.size();
            mcts.push_back(node(next_state));
        }

        double v = search(mcts[x].son[best_act]);
        
        int i = best_act;
        mcts[x].Qsa[i] = (mcts[x].Nsa[i] * mcts[x].Qsa[i] + v) / (mcts[x].Nsa[i] + 1);
        mcts[x].Nsa[i] ++;
        mcts[x].Ns ++;

        return -v;
    }
};

PYBIND11_MODULE(libcpp, m) {
    py::class_<Game>(m, "Game")
        .def(py::init<int, int>())
        .def("getInitBoard", &Game::getInitBoard)
        .def("getBoardSize", &Game::getBoardSize)
        .def("getActionSize", &Game::getActionSize)
        .def("getNextState", &Game::getNextState)
        .def("getValidMoves", &Game::getValidMoves)
        .def("getGameEnded", &Game::getGameEnded)
        .def("getCanonicalForm", &Game::getCanonicalForm)
        .def("stringRepresentation", &Game::stringRepresentation)
        .def("display", &Game::display)
        .def("getFeatureSize", &Game::getFeatureSize)
        .def("getFeature", &Game::getFeature);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const Game &, const py::object, int, double>())
        .def("reset", &MCTS::reset)
        .def("getActionProb", &MCTS::getActionProb)
        .def("getExpertProb", &MCTS::getExpertProb)
        .def("processResult", &MCTS::processResult)
        .def("findLeafToProcess", &MCTS::findLeafToProcess);
}