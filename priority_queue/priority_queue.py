import heapq


class priority_queue(object):

    def __init__(self):
        print('priority_queue')
        self.elements = []

    def __str__(self):
        return ' '.join([str(i) for i in self.elements])

    def __iter__(self):
        for node, key in self.elements:
            yield node

    def top_key(self):
        min_ = heapq.nsmallest(1,  [e[1][0] for e in self.elements])
        for point, keys in self.elements:
            if keys[0] == min_[0]:
                return [(point, keys)]

    def pop(self):
        point = heapq.heappop(self.elements)
        return point[0]

    def insert(self, point, cost):
        heapq.heappush(self.elements, (point, cost))

    def update(self, node, cost):
        self.elements = [e if e[0] != node else (node, cost) for e in self.elements]
        # heapq.heapify(self.elements)

    def remove(self, node):
        self.elements = [e for e in self.elements if e[0] != node]
        # heapq.heapify(self.elements)





