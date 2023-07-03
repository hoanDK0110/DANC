import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt


num_episodes=1000
learning_rate=0.1
discount_factor=0.9
exploration_rate=0.2

PHY = nx.Graph()

# Thêm các nút và trọng số nút
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
SFC.add_node("B", weight=2)
SFC.add_node("C", weight=3)

SFC.add_edge("A", "B", weight=0.05)
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
        
        self.state_space = list(range(len(self.PHY_nodes)))
        self.action_space = list(range(len(self.SFC_nodes)))
        
    def dijkstra(graph, start, end, weight_requirement):
        num_nodes = len(graph)
        distances = np.full(num_nodes, np.inf)  # Khởi tạo khoảng cách ban đầu là vô cùng
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
    q_table = np.zeros((len(env.state_space), len(env.action_space)))
    # Quá trình training
    for episode in range(num_episodes):
        env.reset()
        current_state_space = env.state_space.copy()
        current_action_space = np.array([])
        while 1:
            if len(current_state_space) == 0: # Check xem đã hết trạng thái chưa
                break 
            # Chọn trạng thái
            current_state = current_state_space.pop(0) # Lấy trạng thái từ trong mảng
            pre_state_array = np.append(np.array([]), current_state) # lưu trạng thái vừa chọn vào mảng để tính toán reward
            # Chọn hành động
            while 1:
                 # Khám phá
                if np.random.uniform(0, 1) < exploration_rate:
                    action = np.random.choice(env.action_space) # Chọn hành động bất kì trong không gian hành động
                # Khai thác
                else:
                    action = np.argmax(q_table[current_state]) # Chọn hành động có giá trị Q lớn nhất trong bảng Q table
                
                if action not in current_action_space: # Kiểm tra hành động đó có được chọn trước đo hay chưa
                    current_action_space = np.append(current_action_space,action)
                    break
            
            # Công thức hàm học Q-Learning để cập nhật Q table
            q_table[current_state, action] += learning_rate * (reward + discount_factor * np.max(q_table[current_state_space[0]]) - q_table[current_state, action])
    return q_table

env = SFCMappingEnvironment()
#action = np.random.choice(env.action_space)
#action = env.state_space
#q_table = np.zeros((len(env.action_space), len(env.action_space)))
#current_state_space = env.state_space.copy()
#current_state = current_state_space.pop(0)
#print("Node array:", env.PHY_nodes)
#print("Array weight node:", env.PHY_weights_node)
#print("Edge array:", env.PHY_edges)
#print("Array weight adge:", env.PHY_weight_edges)
#print(env.PHY_array)
