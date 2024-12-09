import pygame
import threading
import time
import random
import math
import heapq
from collections import defaultdict

# Настройки Pygame
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 20
FPS = 60

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 255)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)

class SelectiveRepeatProtocol:
    def __init__(self, node, window_size=4, timeout=2):
        self.node = node
        self.window_size = window_size
        self.timeout = timeout
        self.lock = threading.Lock()
        self.send_base = 0
        self.next_seq_num = 0
        self.buffer = {}
        self.acknowledged = set()
        self.timers = {}
        self.receiver_buffer = defaultdict(dict)

    def send(self, content, destination, network):
        with self.lock:
            if self.next_seq_num < self.send_base + self.window_size:
                message = {
                    'type': 'data',
                    'source': self.node.node_id,
                    'destination': destination,
                    'seq_num': self.next_seq_num,
                    'content': content
                }
                self.buffer[self.next_seq_num] = message
                network.send_message(self.node.node_id, destination, message)
                self.start_timer(self.next_seq_num, network)
                self.next_seq_num += 1

    def start_timer(self, seq_num, network):
        if seq_num in self.timers:
            self.timers[seq_num].cancel()
        self.timers[seq_num] = threading.Timer(self.timeout, self.timeout_retransmit, args=[seq_num, network])
        self.timers[seq_num].start()

    def receive_ack(self, ack):
        with self.lock:
            seq_num = ack['seq_num']
            if seq_num in self.buffer:
                self.acknowledged.add(seq_num)
                if seq_num in self.timers:
                    self.timers[seq_num].cancel()
                    del self.timers[seq_num]
                while self.send_base in self.acknowledged:
                    del self.buffer[self.send_base]
                    self.acknowledged.remove(self.send_base)
                    self.send_base += 1

    def receive_message(self, message, network):
        if message['type'] == 'data':
            self.node.receive_packet(message, network)
        elif message['type'] == 'ack':
            self.receive_ack(message)

    def timeout_retransmit(self, seq_num, network):
        with self.lock:
            if seq_num in self.buffer and seq_num not in self.acknowledged:
                network.send_message(self.node.node_id, self.buffer[seq_num]['destination'], self.buffer[seq_num])
                self.start_timer(seq_num, network)

class Node:
    def __init__(self, node_id, x, y):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.neighbors = []
        self.routing_table = {}
        self.protocol = SelectiveRepeatProtocol(self)

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            self.routing_table[neighbor.node_id] = neighbor.node_id

    def remove_neighbor(self, neighbor):
        if neighbor in self.neighbors:
            self.neighbors.remove(neighbor)
            if neighbor.node_id in self.routing_table:
                del self.routing_table[neighbor.node_id]

    def update_routing_table(self, network):
        distances = {self.node_id: 0}
        previous = {}
        unvisited = set(network.nodes.keys())

        heap = []
        heapq.heappush(heap, (0, self.node_id))

        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_node not in unvisited:
                continue
            unvisited.remove(current_node)

            for neighbor in network.nodes[current_node].neighbors:
                neighbor_id = neighbor.node_id
                distance = current_distance + 1
                if neighbor_id not in distances or distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    previous[neighbor_id] = current_node
                    heapq.heappush(heap, (distance, neighbor_id))

        self.routing_table.clear()
        for destination in network.nodes:
            if destination == self.node_id:
                continue
            if destination in distances:
                path = []
                current = destination
                while current != self.node_id:
                    path.append(current)
                    if current in previous:
                        current = previous[current]
                    else:
                        break
                path.reverse()
                if path:
                    self.routing_table[destination] = path[0]

    def send_message(self, destination_id, content, network):
        if destination_id not in self.routing_table:
            return
        self.protocol.send(content, destination_id, network)

    def receive_packet(self, message, network):
        if message['destination'] == self.node_id:
            ack = {
                'type': 'ack',
                'source': self.node_id,
                'destination': message['source'],
                'seq_num': message['seq_num']
            }
            network.send_message(self.node_id, message['source'], ack)

class Network:
    def __init__(self):
        self.nodes = {}
        self.lock = threading.Lock()

    def add_node(self, node):
        with self.lock:
            self.nodes[node.node_id] = node
            self.update_all_routing_tables()

    def remove_node(self, node_id):
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                for neighbor in list(node.neighbors):
                    self.disconnect_nodes(node_id, neighbor.node_id)
                del self.nodes[node_id]
                self.update_all_routing_tables()

    def connect_nodes(self, node1_id, node2_id):
        with self.lock:
            if node1_id in self.nodes and node2_id in self.nodes:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                node1.add_neighbor(node2)
                node2.add_neighbor(node1)
                self.update_all_routing_tables()

    def disconnect_nodes(self, node1_id, node2_id):
        with self.lock:
            if node1_id in self.nodes and node2_id in self.nodes:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                node1.remove_neighbor(node2)
                node2.remove_neighbor(node1)
                self.update_all_routing_tables()

    def update_all_routing_tables(self):
        for node in self.nodes.values():
            node.update_routing_table(self)

    def send_message(self, source_id, destination_id, message):
        with self.lock:
            if source_id in self.nodes and destination_id in self.nodes:
                source_node = self.nodes[source_id]
                destination_node = self.nodes[destination_id]
                threading.Thread(target=self.deliver_message, args=(source_node, destination_node, message)).start()

    def deliver_message(self, source_node, destination_node, message):
        time.sleep(random.uniform(0.5, 1.5))
        destination_node.protocol.receive_message(message, self)

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Симуляция сети")
    clock = pygame.time.Clock()

    network = Network()
    node_A = Node('A', 100, 300)
    node_B = Node('B', 400, 150)
    node_C = Node('C', 400, 450)
    node_D = Node('D', 700, 300)

    network.add_node(node_A)
    network.add_node(node_B)
    network.add_node(node_C)
    network.add_node(node_D)

    network.connect_nodes('A', 'B')
    network.connect_nodes('B', 'C')
    network.connect_nodes('C', 'D')

    running = True
    while running:
        clock.tick(FPS)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Отрисовка соединений
        for node in network.nodes.values():
            for neighbor in node.neighbors:
                pygame.draw.line(screen, GRAY, (node.x, node.y), (neighbor.x, neighbor.y), 2)

        # Отрисовка узлов
        for node in network.nodes.values():
            pygame.draw.circle(screen, BLUE, (node.x, node.y), NODE_RADIUS)
            font = pygame.font.SysFont(None, 24)
            img = font.render(node.node_id, True, BLACK)
            screen.blit(img, (node.x - img.get_width() // 2, node.y - img.get_height() // 2))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
