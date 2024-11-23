import heapq  # Priority Queue (min-heap) to store open set nodes
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
# Position class to represent (x, y) coordinates
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"({self.x}, {self.y})"

class Cell:
    def __init__(self, position, parent=None, status='free', g=0, h=0, f=0):
        self.position = position  # Position object (x, y)
        self.parent = parent  # Parent cell in the path
        self.status = status  # {'free', 'start', 'dirty', 'clean'}
        self.g = g  # Cost from start to this cell
        self.h = h  # Heuristic cost (estimated)
        self.f = f  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

#Hàm tính khoảng cách Chebyhev giữa 2 ô
def chebyshev_distance(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

# Chia ma trận ra thành 4 phần và đếm số ô dơ của mỗi phần 
def divide_grid_and_count_dirty_cells(grid, dirty_cells):
    rows = len(grid)
    cols = len(grid[0])

    
    top_left = []
    top_right = []
    bottom_left = []
    bottom_right = []

    # Phân loại ô dơ vào từng phần của ma trận
    for cell in dirty_cells:
        (x, y) = cell
        if x < rows // 2 and y < cols // 2:
            top_left.append(cell)
        elif x < rows // 2 and y >= cols // 2:
            top_right.append(cell)
        elif x >= rows // 2 and y < cols // 2:
            bottom_left.append(cell)
        else:
            bottom_right.append(cell)


    # Trả về các phần cùng số lượng ô dơ tương ứng
    return top_left, top_right, bottom_left, bottom_right



# Heuristic Greedy Algorithm để dọn dẹp ô dơ
def greedy_clean(grid, robot_start, dirty_cells,g):
    robot_pos = robot_start 
    total_cost = g
    cleaned_cells = set()
    move = g

    dirty_cells_copy = dirty_cells.copy()

    while dirty_cells_copy:
        # Tìm ô dơ gần nhất
        closest_dirty_cell = min(dirty_cells_copy, key=lambda cell: chebyshev_distance(robot_pos, cell))

        # Tính toán chi phí dọn dẹp
        move_cost = chebyshev_distance(robot_pos, closest_dirty_cell)
        total_cost += move_cost  
        move += move_cost

        #Cập nhật chi phí dọn ô dơ 
        cleaning_dirty_cells_cost = move + 1
        total_cost += cleaning_dirty_cells_cost


        cleaned_cells.add(closest_dirty_cell)
        dirty_cells_copy.remove(closest_dirty_cell)
        robot_pos = closest_dirty_cell
    return total_cost

# Heuristic chia ma trận để dọn dẹp ô dơ
def divide_grid_clean(grid, robot_start, dirty_cells,g):
    robot_pos = robot_start
    total_cost = g
    cleaned_cells = set()
    move = g

    dirty_cells_copy = dirty_cells.copy()

    # Chia ma trận thành 4 phần
    top_left, top_right, bottom_left, bottom_right = divide_grid_and_count_dirty_cells(grid, dirty_cells_copy)



    while dirty_cells_copy:
        # Chọn phần có ô dơ nhiều nhất
        quadrant_dirty_counts = {
            "top_left": len(top_left),
            "top_right": len(top_right),
            "bottom_left": len(bottom_left),
            "bottom_right": len(bottom_right)
        }
        best_quadrant = max(quadrant_dirty_counts, key=quadrant_dirty_counts.get)

        # Chọn phần ma trận có nhiều ô dơ nhất để di chuyển
        if best_quadrant == "top_left":
            target_cells = top_left
        elif best_quadrant == "top_right":
            target_cells = top_right
        elif best_quadrant == "bottom_left":
            target_cells = bottom_left
        else:
            target_cells = bottom_right

        while target_cells:
            closest_dirty_cell = min(target_cells, key=lambda cell: chebyshev_distance(robot_pos, cell))

            # Tính toán chi phí dọn dẹp
            move_cost = chebyshev_distance(robot_pos, closest_dirty_cell)
            total_cost += move_cost 

            move += move_cost

            # Cập nhật chi phí dọn dẹp
            cleaning_dirty_cells_cost = move + 1
            total_cost += cleaning_dirty_cells_cost

            #Cập nhật trạng thái và vị trí mới cho các ô được xử lý
            cleaned_cells.add(closest_dirty_cell)
            dirty_cells_copy.remove(closest_dirty_cell)
            target_cells.remove(closest_dirty_cell)
            robot_pos = closest_dirty_cell
            
        # Cập nhật các phần ma trận sau khi dọn 
        top_left, top_right, bottom_left, bottom_right = divide_grid_and_count_dirty_cells(grid, dirty_cells_copy)

    return total_cost

# Thuật toán A* tìm đường đi 
def a_star(grid, robot_start, dirty_cells):
    open_list = [] #Danh sach các ô cần xét (sử dụng heap)
    move = 0
    cleaning_cost = 1

    # Node start
    start_cell = Cell(position=robot_start, status='start', g=0, h=0, f=0)
    robot_start_tuple = (robot_start.x, robot_start.y)
    start_cell.h = min(greedy_clean(grid, robot_start_tuple, dirty_cells,g = 0), divide_grid_clean(grid, robot_start_tuple, dirty_cells,g = 0))
    start_cell.f = start_cell.g + start_cell.h

    # Thêm start node vào open list
    heapq.heappush(open_list, start_cell)

    while open_list:
        current_cell = heapq.heappop(open_list)
        open_list.clear()
        current_pos = current_cell.position
        if (current_pos.x,current_pos.y) in dirty_cells:
            current_cell.g += (cleaning_cost)
            dirty_cells.remove((current_pos.x,current_pos.y))
        # Nếu không còn ô dơ thì truy lại đường đi
        if not dirty_cells:
            path = []
            final_cell = current_cell
            while current_cell:
                path.append(current_cell.position)
                current_cell = current_cell.parent
            path.reverse()
            return path, final_cell.g
        neighbors = [
            Position(current_pos.x - 1, current_pos.y),  # Up
            Position(current_pos.x + 1, current_pos.y),  # Down
            Position(current_pos.x, current_pos.y - 1),  # Left
            Position(current_pos.x, current_pos.y + 1),   # Right
            Position(current_pos.x - 1, current_pos.y + 1),  # Up right
            Position(current_pos.x - 1, current_pos.y - 1),  # Up left
            Position(current_pos.x + 1, current_pos.y + 1),  # down right
            Position(current_pos.x + 1, current_pos.y - 1)  # down left
        ]

        for neighbor in neighbors:
            if not (0 <= neighbor.x < len(grid) and 0 <= neighbor.y < len(grid[0])):
                continue

            neighbor_tuple = (neighbor.x, neighbor.y)
            g_cost = current_cell.g + 1  # Assuming all moves cost 1
            h_cost = min(greedy_clean(grid, neighbor_tuple, dirty_cells,g = g_cost), divide_grid_clean(grid, neighbor_tuple, dirty_cells,g = g_cost))
            f_cost = g_cost + h_cost

            neighbor_cell = Cell(position=neighbor, parent=current_cell, status='free', g=g_cost, h=h_cost, f=f_cost)


            heapq.heappush(open_list, neighbor_cell)

        move += 1
        cleaning_cost +=1 

    return None, float('inf')  

def split_into_columns(data, max_items_per_column):

    num_columns = math.ceil(len(data) / max_items_per_column)
    column_size = math.ceil(len(data) / num_columns) 
    return [data[i:i + column_size] for i in range(0, len(data), column_size)]



def visualize(grid_size, dirty_cells, path, cost):
    # Tạo grid với màu sắc theo trạng thái
    grid = np.zeros(grid_size)
    path_x = [pos.x for pos in path]
    path_y = [pos.y for pos in path]

    fig, ax = plt.subplots(figsize=(10,5))

    max_size = min(25, 500 / max(grid_size))  # Giới hạn kích thước lớn nhất
    marker_size = max_size
    ax.imshow(grid, cmap='binary')
    for dirty_cell in dirty_cells:
        ax.plot(dirty_cell[1], dirty_cell[0], 's', markersize=marker_size, color='grey')

    # Vẽ lưới và đường đi
    ax.set_xticks(np.arange(0.5, grid_size[1], 1))
    ax.set_yticks(np.arange(0.5, grid_size[0], 1))
    ax.invert_yaxis()
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xticklabels(np.arange(1, grid_size[1]+1, 1), minor=True)
    ax.set_yticklabels(np.arange(1, grid_size[0]+1, 1), minor=True)
    ax.set_xticks(np.arange(0, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(0, grid_size[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='', linewidth=1)
    ax.grid(which='major', color='red', linestyle='-', linewidth=1)

    # Animation robot
    robot, = ax.plot([], [], 's', markersize=marker_size, color='cyan')
    trail, = ax.plot([], [], 'ro-', markersize=8, linewidth=2, color='red')

    cost_text = ax.text(
        0.25, 1.05,  
        f"Path found with total cost: {cost}",
        ha="left", va="center", color="black", fontsize=10,
        transform=ax.transAxes  # Sử dụng tọa độ tương đối
    )
    path_text_list = [f"{i + 1}: ({pos.x + 1}, {pos.y + 1})" for i, pos in enumerate(path)]

    columns = split_into_columns(path_text_list, max_items_per_column=35)


    for i, column in enumerate(columns):
        # Nối các phần tử trong cột thành chuỗi
        path_text_list = "\n".join(column)
        ax.text(
            -0.6 - i * 0.3 if i < len(columns) // 2 else 1.1 + (i - len(columns) // 2) * 0.3,
            0.5,  # Điều chỉnh vị trí y của tất cả cột để cân đối
            path_text_list,
            ha="left" if i < len(columns) // 2 else "left",
            va="center", color="black", fontsize=10,
            transform=ax.transAxes
        )

    def animate(i):
        # Robot di chuyển và vẽ dấu vết
        robot.set_data(path_y[:i + 1], path_x[:i + 1])
        trail.set_data(path_y[:i + 1], path_x[:i + 1])
        return robot, trail

    ani = FuncAnimation(fig, animate, frames=len(path), interval=500, blit=False, repeat=False)
    plt.show()


def run_simulation(grid_size,num_dirty_cells ):
    #Nhập kích thước ma trận từ người dùng
    while True:
        try:
            rows = int(input("Nhập số hàng của ma trận: "))
            cols = int(input("Nhập số cột của ma trận: "))
            if rows <= 0 or cols <= 0:
                print("Số hàng và số cột phải là số nguyên dương. Vui lòng nhập lại!")
                continue
            grid_size = (rows, cols)
            break
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ!")

    while True:
        try:
            num_dirty_cells = int(input("Nhập số lượng ô bẩn (dirty cells): "))
            if num_dirty_cells < 0 or num_dirty_cells > rows * cols:
                print(f"Số lượng ô bẩn phải nằm trong khoảng từ 0 đến {rows * cols}. Vui lòng nhập lại!")
                continue
            break
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ!")

    # Khởi tạo grid với các Cell (mặc định 'free')
    grid = [[Cell(Position(x, y), status='free') for y in range(grid_size[1])] for x in range(grid_size[0])]

    #Đặt vị trí bắt đầu cho robot
    start_pos = Position(
        random.randint(0, grid_size[0] - 1),
        random.randint(0, grid_size[1] - 1)
    )

    grid[start_pos.x][start_pos.y].status = 'start'

    # Đặt các ô bẩn (dirty)
    dirty_cells = set()
    while len(dirty_cells) < num_dirty_cells:
        dirty_cells.add((random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1)))
    
    for x, y in dirty_cells:
        grid[x][y].status = 'dirty'
    dirty_cells_cop = dirty_cells.copy()
    print("Dirty Cells:", dirty_cells)
    print("Start Position:", start_pos)

    # Chạy thuật toán A*
    path, cost = a_star(grid, start_pos, dirty_cells)

    # Cập nhật trạng thái ô khi đã được dọn
    if path:
        for pos in path:
            x, y = pos.x, pos.y
            if (x, y) in dirty_cells:
                grid[x][y].status = 'clean'
        print(f"Path found with total cost: {cost}")
        print("Path:", [str(p) for p in path])
        visualize(grid_size, dirty_cells_cop, path, cost)
    else:
        print("No path found.")

run_simulation(grid_size=(10,10), num_dirty_cells=10)


