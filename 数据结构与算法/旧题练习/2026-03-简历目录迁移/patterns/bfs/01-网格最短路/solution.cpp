#include <iostream>
#include <queue>
#include <vector>
using namespace std;

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

int main()
{
    int n, m;
    cin >> n >> m;
    int sx, sy, tx, ty;
    vector<vector<char>> grid(n + 1, vector<char>(m + 1, '#'));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            char ch;
            cin >> ch;
            grid[i][j] = ch;
            if (ch == 'S')
            {
                sx = i;
                sy = j;
            }
            if (ch == 'T')
            {
                tx = i;
                ty = j;
            }
        }
    }
    int step = 0;
    queue<pair<int, int>> que;
    vector<vector<int>> vis(n + 1, vector<int>(m + 1, 0));
    vis[sx][sy] = 1;
    que.push({sx, sy});
    while (!que.empty())
    {
        int sz = que.size();
        while (sz--)
        {
            pair<int, int> cur = que.front();
            que.pop();
            int x = cur.first, y = cur.second;
            if (x == tx && y == ty)
            {
                cout << step << endl;
                return 0;
            }
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx >= 1 && nx <= n && ny >= 1 && ny <= m && grid[nx][ny] != '#' && !vis[nx][ny])
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
