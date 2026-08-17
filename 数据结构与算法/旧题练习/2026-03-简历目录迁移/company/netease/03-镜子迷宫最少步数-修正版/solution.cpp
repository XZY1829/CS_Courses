#include <iostream>
#include <queue>
#include <vector>


using namespace std;



//up0 down1 left2 right3
int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

//  '/'
int change1(int d)
{
    if (d == 0) return 3;
    if (d == 1) return 2;
    if (d == 2) return 1;
    return 0;
}

// '\'
int change2(int d)
{
    if (d == 0) return 2;
    if (d == 1) return 3;
    if (d == 2) return 0;
    return 1;
}


int main()
{

    int m, n, k, sr, sc, er, ec;
    cin >> m >> n;
    cin >> k;
    const int mm = m + 2, nn = n + 2;

    vector<vector<char>> grid(mm, vector<char>(nn, '#'));
    for (int i = 1; i <= m; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            grid[i][j] = '.';
        }
    }
    for (int i = 0; i < k; i++)
    {
        int r, c;
        char ch;
        cin >> r >> c >> ch;
        grid[r][c] = ch;
    }
    cin >> sr >> sc;
    cin >> er >> ec;
    vector<vector<int>> vis(mm, vector<int>(nn, 0));
    queue<pair<int, int>> que;
    int step = 0;
    que.push({sr, sc});
    vis[sr][sc] = 1;
    while (!que.empty())
    {
        int sz = que.size();
        while (sz--)
        {
            auto cur = que.front();
            que.pop();
            int x = cur.first;
            int y = cur.second;
            for (int i = 0; i < 4; i++)
            {
                int dir = i;
                int nx = x;
                int ny = y;
                while (true)
                {
                    int tx = nx + dx[dir];
                    int ty = ny + dy[dir];

                    if (grid[tx][ty] == '#')
                    {
                        break;
                    }

                    nx = tx;
                    ny = ty;

                    if (tx == er && ty == ec)
                    {
                        cout << step + 1 << endl;
                        return 0;
                    }

                    if (grid[tx][ty] == '/')
                    {
                        int ndir = change1(dir);

                        int nnx = tx + dx[dir];
                        int nny = ty + dy[dir];

                        if (grid[nnx][nny] == '#' || grid[nnx][nny] == '/' || grid[nnx][nny] == '\\')
                        {
                            nx -= dx[dir];
                            ny -= dy[dir];
                            break;
                        }
                        dir = ndir;
                    }
                    else if (grid[tx][ty] == '\\')
                    {
                        int ndir = change2(dir);

                        int nnx = tx + dx[ndir];
                        int nny = ty + dy[ndir];

                        if (grid[nnx][nny] == '#' || grid[nnx][nny] == '/' || grid[nnx][nny] == '\\')
                        {
                            nx -= dx[dir];
                            ny -= dy[dir];
                            break;
                        }
                        dir = ndir;
                    }
                }
                if (vis[nx][ny] == 0)
                {
                    vis[nx][ny] = 1;
                    que.push({nx, ny});
                }
            }
        }
        step++;
    }
    cout << -1 << endl;
}
