from heapq import heappush, heappop

def expand_row(row):
    result = []
    i = 0
    while i < len(row):
        j = i
        while j < len(row) and row[j].isdigit():
            j += 1
        result.extend([row[j]] * int(row[i:j]))
        i = j + 1
    return result

N = int(input())
grid = [expand_row(input().strip()) for _ in range(N)]

for r in range(N):
    for c in range(N):
        if grid[r][c] == 'S': start = (r, c)
        if grid[r][c] == 'D': end = (r, c)

pq = [(0, start[0], start[1])]
visited = [[float('inf')]*N for _ in range(N)]
visited[start[0]][start[1]] = 0
dirs = [(-1,0),(1,0),(0,-1),(0,1)]

while pq:
    broken, r, c = heappop(pq)
    if (r, c) == end:
        print(broken)
        break
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != 'R':
            nb = broken + (grid[nr][nc]=='G')
            if nb < visited[nr][nc]:
                visited[nr][nc] = nb
                heappush(pq, (nb, nr, nc))
