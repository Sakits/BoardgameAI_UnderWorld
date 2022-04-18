#include "Game.hpp"
namespace py = pybind11;

Game::Game(int _n, int _win_n) : n(_n), win_n(_win_n) 
{
    init();
}

void Game::init()
{
    board.resize(n);
    for (int i = 0; i < n; i++)
    {
        board[i].resize(n);
        for (int j = 0; j < n; j++)
            board[i][j] = 0;
    }

    valids.resize(n * n + 1);
    for (int i = 0; i < n * n; i++)
        valids[i] = 1;
    valids[n * n] = 0;
}

void Game::get_next_state(int action)
{
    if (action == n * n)
    {
        now_player = - now_player;
        return;
    }

    int x = action / n, y = action % n;
    assert(board[x][y] == 0);
    board[x][y] = now_player;
    now_player = -now_player;
}

void Game::get_valid_moves()
{
    int cnt = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            valids[i * n + j] = !board[i][j], cnt += !board[i][j];
    valids[n * n] = !cnt;
}

double Game::get_game_ended()
{
    int cnt = n * n;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        if (board[i][j])
        {
            cnt--;

            int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
            for (int k = 0; k < win_n; k++)
            {
                cnt1 += i + k < n && board[i][j] == board[i + k][j];
                cnt2 += j + k < n && board[i][j] == board[i][j + k];
                cnt3 += i + k < n && j + k < n && board[i][j] == board[i + k][j + k];
                cnt4 += i - k >= 0 && j + k < n && board[i][j] == board[i - k][j + k];
            }

            if (cnt1 == win_n || cnt2 == win_n || cnt3 == win_n || cnt4 == win_n)
                return board[i][j];
        }
    }

    return cnt ? 0 : 1e-4;
}

void Game::get_canonical_form()
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            board[i][j] *= now_player;
    
    now_player = 1;
}

std::string Game::string_representation()
{
    std::string s = "";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += board[i][j] == 1 ? "w" : (board[i][j] == -1 ? "b" : " ");

    return s;
}

/* -------------------------- py API --------------------------*/
void Game::get_board(py::array_t<char> pyboard, int player)
{
    char* ptr = static_cast<char*>(pyboard.request().ptr);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            board[i][j] = ptr[i * n + j];
    
    now_player = player;
}

py::array_t<char> Game::return_board() 
{
    auto pyboard = py::array_t<char>(n * n);
    char* ptr = static_cast<char *>(pyboard.request().ptr);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            ptr[i * n + j] = board[i][j];

    pyboard.resize({n, n});
    return pyboard;
}

py::array_t<char> Game::getInitBoard()
{
    init();

    return return_board();
}

py::tuple Game::getBoardSize()
{
    return py::make_tuple(n, n);
}

int Game::getActionSize()
{
    return n * n + 1;
}

py::tuple Game::getNextState(py::array_t<char> pyboard, int player, int action)
{
    get_board(pyboard, player);

    get_next_state(action);

    return py::make_tuple(return_board(), now_player);
}

py::array_t<char> Game::getValidMoves(py::array_t<char> pyboard, int player)
{
    get_board(pyboard, player);

    get_valid_moves();

    auto pyvalids = py::array_t<char>(n * n + 1);
    char* ptr = static_cast<char *>(pyvalids.request().ptr);

    for (int i = 0; i < n * n + 1; i++)
        ptr[i] = valids[i];

    return pyvalids;
}

double Game::getGameEnded(py::array_t<char> pyboard, int player)
{
    get_board(pyboard, player);

    return get_game_ended();
}

py::array_t<char> Game::getCanonicalForm(py::array_t<char> pyboard, int player)
{
    get_board(pyboard, player);

    get_canonical_form();

    return return_board();
}

void Game::display(py::array_t<char> pyboard)
{
    get_board(pyboard, 1);

    for (int i = 0; i < n; i++)
        printf("%d |", i);
    puts("");
    puts(" -----------------------");
    for (int i = 0; i < n; i++)
    {
        printf("%d |", i);
        for (int j = 0; j < n; j++)
            printf("%s ", board[i][j] == 1 ? "w" : (board[i][j] == -1 ? "b" : " "));
        puts("|");
    }
    puts(" -----------------------");
}

std::string Game::stringRepresentation(py::array_t<char> pyboard)
{
    get_board(pyboard, 1);
    return string_representation();
}


