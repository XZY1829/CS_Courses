#include <iostream>
#include <queue>
#include <vector>
using namespace std;

//up0 down1 left2 right3
int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

//  /
int change1(int dir)
{
    if (dir == 0) return 3;
    if (dir == 1) return 2;
    if (dir == 2) return 1;
    return 0;
}

// '\\'
int change2(int dir)
{
    if (dir == 0) return 2;
    if (dir == 1) return 3;
    if (dir == 2) return 0;
    return 1;
}

int main()
{
    int n, m, k;
    cin >> n >> m >> k;
    int sr, sc, er, ec;
    cin >> sr >> sc >> er >> ec;
    vector<vector<char>> grid(n + 1, vector<char>(m + 1, '#'));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            grid[i][j] = '.';
        }
    }
    for (int i = 0; i < k; i++)
    {
        int x, y;
        cin >> x >> y;
        char ch;
        cin >> ch;
        grid[x][y] = ch;
    }

    vector<vector<int>> visited(n + 1, vector<int>(m + 1, 0));
    int step = 1;
    queue<pair<int, int>> que;
    que.push({sr, sc});
    visited[sr][sc] = 1;
    while (!que.empty())
    {
        int sz = que.size();
        while (sz--)
        {
            pair<int, int> cur = que.front();
            que.pop();
            int x = cur.first;
            int y = cur.second;
            if (x == er && y == ec)
            {
                cout << step << endl;
                return 0;
            }


            int nx = x;
            int ny = y;
            if (nx < 1 || nx > n || ny < 1 || ny > m || grid[nx][ny] != '#')
            {
                break;
            }
            for (int i = 0; i < 4; i++)
            {
                int dir = i;
                nx = x + dx[dir];
                ny = y + dy[dir];
                while (true)
                {
                    if (nx == er && ny == ec)
                    {
                        cout << step << endl;
                        return 0;
                    }
                    if (nx < 1 || nx > n || ny < 1 || ny > m || grid[nx][ny] == '#')
                    {
                        nx -= dx[dir];
                        ny -= dy[dir];
                        break;
                    }
                    if (grid[nx][ny] == '/')
                    {
                        dir = change1(dir);
                        nx += dx[dir];
                        ny += dy[dir];
                        if (nx < 1 || nx > n || ny < 1 || ny > m || grid[nx][ny] == '#')
                        {
                            nx -= dx[dir];
                            ny -= dy[dir];
                            break;
                        }
                    }
                    if (grid[nx][ny] == '\\')
                    {
                        dir = change2(dir);
                        nx += dx[dir];
                        ny += dy[dir];
                        if (nx < 1 || nx > n || ny < 1 || ny > m || grid[nx][ny] == '#')
                        {
                            nx -= dx[dir];
                            ny -= dy[dir];
                            break;
                        }
                    }
                    if (grid[nx][ny] == '.')
                    {
                        visited[nx][ny] = 1;
                        nx += dx[dir];
                        ny += dy[dir];
                        if (nx < 1 || nx > n || ny < 1 || ny > m || grid[nx][ny] == '#')
                        {
                            nx -= dx[dir];
                            ny -= dy[dir];
                            break;
                        }
                    }
                }
                que.push({nx, ny});
            }
            step++;
        }
    }

    cout << -1 << endl;
    return 0;
}
