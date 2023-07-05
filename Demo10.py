import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt


num_episodes=1000
learning_rate=0.1
discount_factor=0.9
exploration_rate=0.2



# Thêm các nút và trọng số nút
PHY = nx.Graph()
PHY.add_node("0", weight=5)
PHY.add_node("1", weight=2)
PHY.add_node("2", weight=4)
PHY.add_node("3", weight=1)
PHY.add_node("4", weight=3)
PHY.add_node("5", weight=2)

PHY.add_edge("0", "1", weight=0.6)
PHY.add_edge("0", "2", weight=0.2)
PHY.add_edge("2", "3", weight=0.1)
PHY.add_edge("2", "4", weight=0.7)
PHY.add_edge("2", "5", weight=0.9)
PHY.add_edge("0", "3", weight=0.3)


# Thêm các cạnh và trọng số cạnh
SFC = nx.Graph()
SFC.add_node("A", weight=1)
SFC.add_node("B", weight=1)
SFC.add_node("C", weight=1)

SFC.add_edge("A", "B", weight=0.1)
SFC.add_edge("B", "C", weight=0.2)


class SFCMappingEnvironment:
    def __init__(self):
        # Thiết lập mạng vật lý
        self.PHY_nodes = list(PHY.nodes())
        self.PHY_weights_node = [PHY.nodes[node]['weight'] for node in self.PHY_nodes]
        self.PHY_edges = list(PHY.edges())
        self.PHY_weight_edges = [PHY.edges[edge]['weight'] for edge in self.PHY_edges]
        self.PHY_array = nx.adjacency_matrix(PHY).toarray()
        
        # Thiết lập mạng SFC
        self.SFC_nodes = list(SFC.nodes())
        self.SFC_weights_node = [SFC.nodes[node]['weight'] for node in self.SFC_nodes]
        self.SFC_edges = list(SFC.edges())
        self.SFC_weight_edges = [SFC.edges[edge]['weight'] for edge in self.SFC_edges]
        self.SFC_array = nx.adjacency_matrix(SFC).toarray()

        # Thiết lập không gian trạng thái và hành động
        
        self.state_space = list(range(len(self.SFC_nodes)))
        self.action_space = list(range(len(self.PHY_nodes)))
    
   # def calculate_reward(self, PHY_node_start, PHY_node_end, SFC_node_start, SFC_node_end):
        #if self.PHY_weights_node[PHY_node_start] > self.SFC_weights_node[SFC_node_start] and self.PHY_weights_node[PHY_node_end] > self.SFC_weights_node[SFC_node_end]:
        
def dijkstra(graph, start, end, weight_requirement):
    num_nodes = len(graph)
    print("num_node: ", num_nodes)
    print("end: ", end)
    distances = np.full(num_nodes, np.inf)  # Khởi tạo khoảng cách ban đầu là vô cùng
    print("start: ", start)
    distances[start] = 0  # Khoảng cách từ nút bắt đầu đến chính nó là 0

    visited = set()  # Tập các nút đã được duyệt
    previous = np.full(num_nodes, None)  # Mảng lưu các nút trước đó trên đường đi ngắn nhất

    # Duyệt qua tất cả các nút
    for _ in range(num_nodes):
        # Tìm nút có khoảng cách nhỏ nhất và chưa được duyệt
        min_distance = np.inf
        min_node = None
        for node in range(num_nodes):
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        if min_node is None:
            break  # Không có đường đi từ nút bắt đầu đến nút kết thúc

        visited.add(min_node)  # Đánh dấu nút đã được duyệt

        # Kiểm tra nếu đã đến nút kết thúc
        if min_node == end:
            path = []
            node = end
            while node is not None:
                path.insert(0, node)
                node = previous[node]
            return len(path), path  # Trả về số lượng nút phải đi qua và đường đi

        # Cập nhật khoảng cách và nút trước đó cho các nút kề
        for neighbor in range(num_nodes):
            if neighbor not in visited and graph[min_node][neighbor] >= weight_requirement:
                new_distance = distances[min_node] + graph[min_node][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = min_node

    return -1, []  # Không tìm thấy đường đi từ nút bắt đầu đến nút kết thúc
      


# Thuật toán Q-Learning
def q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate):
    # Khởi tạo bảng Q table là ma trận 3(số node SFC) x 6(số node PHY)
    #q_table = np.zeros((len(env.state_space), len(env.action_space)))
    
    rows = 3
    cols = 6
    start_value = 1
    q_table = [[start_value + (cols * row) + col for col in range(cols)] for row in range(rows)]

    # Quá trình training
    for episode in range(num_episodes):
        current_state_space = env.state_space.copy()
        print("current_state_space: ",current_state_space)
        selected_action_space = np.array([])
        selected_state_array = np.array([])
        print("selected_action_space: ",selected_action_space)
        while 1:
            if len(current_state_space) == 0: # Check xem đã hết trạng thái chưa
                break 
            # Chọn trạng thái
            current_state = current_state_space.pop(0) # Lấy trạng thái từ trong mảng
            print("current_state: ",current_state)
            selected_state_array = np.append(selected_state_array, current_state)# lưu trạng thái vừa chọn vào mảng để tính toán reward
            print("selected_state_array: ",selected_state_array)
            
            # Chọn hành động (chọn node PHY) thỏa mãn cap of PHY node > cap of SFC node và link map giữa 2 node PHY > link giữa 2 SFC liên tiếp
            while 1:
                # Khám phá
                if np.random.rand() < exploration_rate:
                    action = np.random.choice(env.action_space) # Chọn hành động bất kì trong không gian hành động    
                # Khai thác
                else:
                    action = np.argmax(q_table[current_state]) # Chọn hành động có giá trị Q lớn nhất trong bảng Q table
                #check_action(action)
                if action not in selected_action_space and env.PHY_weights_node[action] >= env.SFC_weights_node[current_state]: # Kiểm tra hành động đó có được chọn trước đo hay chưa
                    if len(selected_state_array) == 1: 
                        selected_action_space = np.append(selected_action_space,action)
                        print("selected_action_space: ", selected_action_space[-1])
                        
                        break
                    else:
                        print("selected_action_space: ", selected_action_space[-1])
                        num_hop, path = dijkstra(env.PHY_array, int(selected_action_space[-1])  , action, env.SFC_array[current_state - 1][current_state])
                        print("path: ",path)
                        if num_hop != - 1:
                            selected_action_space = np.append(selected_action_space,action)
                            break
            print("Action:", action)
            if len(selected_state_array) == 1: 
                reward = 100 - 2 * (env.PHY_weights_node[action] - env.SFC_weights_node[current_state])
                if current_state_space:
                    q_table[current_state][action] = (1-learning_rate) * q_table[current_state][action] + learning_rate * (reward + discount_factor * np.max(q_table[current_state_space[0]]))
                else:
                    q_table[current_state][action] = (1-learning_rate) * q_table[current_state][action] + learning_rate * reward
            else:
                reward = 100 - 2 * (env.PHY_weights_node[action] - env.SFC_weights_node[current_state]) - num_hop
                if current_state_space:
                    q_table[current_state][action] = (1-learning_rate) * q_table[current_state][action] + learning_rate * (reward + discount_factor * np.max(q_table[current_state_space[0]]))
                else:
                    q_table[current_state][action] = (1-learning_rate) * q_table[current_state][action] + learning_rate * reward
            
                
    return q_table

env = SFCMappingEnvironment()

q_table = q_learning(env, num_episodes, learning_rate, discount_factor, exploration_rate)

selected_columns = set()

print("Q table co gia tri la:" ,q_table)
# Duyệt qua từng hàng của mảng
#for row in q_table:
    # Tìm vị trí của phần tử lớn nhất trong hàng
#    max_index = np.argmax(row)
    
    # Kiểm tra điều kiện nếu cột 1 đã được chọn thì chọn cột khác
#    while max_index == 1 and max_index in selected_columns:
#       row[max_index] = float('-inf')  # Đánh dấu cột đã chọn để không được chọn lại
#        max_index = np.argmax(row)
    
    # Lưu trữ vị trí của cột đã chọn
#    selected_columns.add(max_index)
    
    # In kết quả
#   print(f"Index max in row: ({max_index}, {np.where(q_table == row[max_index])[0][0]})")


