import numpy as np
from skimage.draw import line
from skimage.measure import approximate_polygon
cimport cython
from cpython.array cimport array, clone

cdef int ravel(int i, int j, int num_cols):
    return i * num_cols + j

cdef class GridGraph:
    cdef unsigned char[:, ::1] grid
    cdef int num_rows
    cdef int num_cols
    cdef int max_num_verts
    cdef int max_edges_per_vert
    cdef float inf
    cdef array int_array_template
    cdef array float_array_template
    cdef array edges
    cdef array edge_counts
    cdef array weights
    cdef dict cache

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    def __init__(self, unsigned char[:, ::1] grid):
        self.grid = grid

        cdef int num_dirs = 8
        cdef int[8][2] dirs = [[0, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1], [1, -1], [1, 0], [1, 1]]
        cdef float sqrt_2 = np.sqrt(2)
        cdef float[8] dir_lengths = [1, 1, sqrt_2, 1, sqrt_2, sqrt_2, 1, sqrt_2]

        self.num_rows = self.grid.shape[0]
        self.num_cols = self.grid.shape[1]
        self.max_num_verts = self.num_rows * self.num_cols
        self.max_edges_per_vert = num_dirs
        self.inf = 2 * self.max_num_verts

        self.int_array_template = array('i')
        self.float_array_template = array('f')
        edges = clone(self.int_array_template, self.max_num_verts * self.max_edges_per_vert, zero=False)
        edge_counts = clone(self.int_array_template, self.max_num_verts, zero=True)
        weights = clone(self.float_array_template, self.max_num_verts * self.max_edges_per_vert, zero=False)

        # Construct graph (edges and weights)
        cdef int i, j
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                v = ravel(i, j, self.num_cols)
                if self.grid[i, j] == 0:
                    continue
                for k in range(num_dirs):
                    ip = i + dirs[k][0]
                    jp = j + dirs[k][1]
                    if (ip < 0 or jp < 0 or ip >= self.num_rows or jp >= self.num_cols or self.grid[ip, jp] == 0):
                        continue
                    e = ravel(v, edge_counts.data.as_ints[v], self.max_edges_per_vert)
                    edges.data.as_ints[e] = ravel(ip, jp, self.num_cols)
                    weights.data.as_floats[e] = dir_lengths[k]
                    edge_counts.data.as_ints[v] += 1

        self.edges = edges
        self.edge_counts = edge_counts
        self.weights = weights

        self.cache = {}

    cdef _spfa(self, int source_i, int source_j):
        # Create arrays for distances and parents
        cdef array dists = clone(self.float_array_template, self.max_num_verts, zero=False)
        cdef array parents = clone(self.int_array_template, self.max_num_verts, zero=False)
        for v in range(self.max_num_verts):
            dists.data.as_floats[v] = self.inf
            parents.data.as_ints[v] = -1

        # Create queue
        cdef array queue = clone(self.int_array_template, self.max_num_verts * self.max_edges_per_vert, zero=False)
        cdef array in_queue = clone(self.int_array_template, self.max_num_verts, zero=True)

        # Run SPFA
        cdef int head = 0, tail = 0
        cdef float new_dist
        s = ravel(source_i, source_j, self.num_cols)
        dists.data.as_floats[s] = 0
        tail += 1
        queue.data.as_ints[tail] = s
        in_queue.data.as_ints[s] = 1
        while head < tail:
            head += 1
            u = queue.data.as_ints[head]
            in_queue.data.as_ints[u] = 0
            for k in range(self.edge_counts.data.as_ints[u]):
                e = ravel(u, k, self.max_edges_per_vert)
                v = self.edges.data.as_ints[e]
                new_dist = dists.data.as_floats[u] + self.weights.data.as_floats[e]
                if new_dist < dists.data.as_floats[v]:
                    parents.data.as_ints[v] = u
                    dists.data.as_floats[v] = new_dist
                    if not in_queue.data.as_ints[v]:
                        tail += 1
                        queue.data.as_ints[tail] = v
                        in_queue.data.as_ints[v] = 1
                        if dists.data.as_floats[queue.data.as_ints[tail]] < dists.data.as_floats[queue.data.as_ints[head + 1]]:
                            tmp = queue.data.as_ints[tail]
                            queue.data.as_ints[tail] = queue.data.as_ints[head + 1]
                            queue.data.as_ints[head + 1] = tmp

        # Fill unreachable locations with -1
        for v in range(self.max_num_verts):
            if dists.data.as_floats[v] >= self.inf - 1e-6:
                dists.data.as_floats[v] = -1

        return dists, parents

    cdef _spfa_with_cache(self, source):
        if source not in self.cache:
            self.cache[source] = self._spfa(source[0], source[1])
        return self.cache[source]

    @cython.cdivision(True)
    def shortest_path(self, source, target):
        cdef array parents
        _, parents = self._spfa_with_cache(source)

        # Recover shortest path (target to source)
        source_i, source_j = source
        target_i, target_j = target
        cdef int num_cols = self.grid.shape[1]
        cdef int u = ravel(source_i, source_j, num_cols)
        cdef int v = ravel(target_i, target_j, num_cols)
        dense_path = [[v / num_cols, v % num_cols]]  # C integer division
        while not v == u:
            v = parents.data.as_ints[v]
            if v < 0:
                break
            dense_path.append([v / num_cols, v % num_cols])  # C integer division

        # Convert dense path to sparse path (waypoints)
        sparse_path = approximate_polygon(np.array(dense_path), tolerance=1)

        # Remove unnecessary waypoints
        path = [sparse_path[0]]
        for k in range(1, sparse_path.shape[0] - 1):
            rr, cc = line(*path[-1], *sparse_path[k + 1])
            if (1 - self.grid.base[rr, cc]).sum() > 0:
                path.append(sparse_path[k])
        if len(sparse_path) > 1:
            path.append(sparse_path[-1])

        # Reverse path
        path = path[::-1]

        return path

    def shortest_path_distance(self, source, target):
        cdef array dists
        dists, _ = self._spfa_with_cache(source)
        cdef int target_i, target_j
        target_i, target_j = target
        cdef int num_cols = self.grid.shape[1]
        cdef int v = ravel(target_i, target_j, num_cols)
        return dists.data.as_floats[v]

    def shortest_path_image(self, source):
        dists, _ = self._spfa_with_cache(source)
        return np.asarray(dists).reshape(self.grid.shape[0], self.grid.shape[1])
