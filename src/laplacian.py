
import numpy as np

def prune_to_triangulation(valid_idx, neighbors, grid, n_patch_per_dim, img_alphas):
    """
    Given the vertex set and edge set of a triangle mesh with potentially overlapping edges, return a corresponding triangulation without overlapping edges.
    
    Overlapping edges exist if two neighbors B and C of node A are connected and their lists of neighbors have an intersection D also in the neighborhood of 
    A. The overlapping edges are then B-C and A-D. Keep the edge that satisfies the Delaunay criteria (sum of angles for the edge < pi)
    """
    out_of_bounds_entry = (-1, -1)

    # TODO(as) debug. ensure all neighborhood relations are symmetric
    # for n in neighbors:
    #     for ni in neighbors[n]:
    #         if ni != out_of_bounds_entry:
    #             if n not in neighbors[ni]:
    #                 print("ugh")

    for idx in valid_idx:
        nbrhood = neighbors[idx]

        if len(nbrhood) == 0:
            print("odd")
        
        # find a pair of neighbors that are also neighbors
        for b in nbrhood:
            if b == out_of_bounds_entry:
                continue

            for c in nbrhood:
                if b == c or c == out_of_bounds_entry:
                    continue
                if b not in neighbors[c]:
                    assert c not in neighbors[b]        # should be symmetric
                    continue
                if c not in neighbors[b]:
                    print("ugh")
                assert c in neighbors[b]        # should be symmetric
                
                # b and c are neighbors. Find intersection of their neighbor sets (minus idx)
                intersection = set(neighbors[b]) & set(neighbors[c])
                assert idx in intersection

                # check if this intersection includes any other neighbors of idx
                intersection = set(nbrhood) & intersection

                # TODO(as) prune B-C or A-D
                # NOTE: prune B-C for now since it works better with these loops (does not edit nbrhood)
                if len(intersection) > 0:
                    # neighbors[b].remove(c)
                    # neighbors[c].remove(b)
                    neighbors[b] = [i for i in neighbors[b] if i != c]
                    neighbors[c] = [i for i in neighbors[c] if i != b]


                    # for n in neighbors:
                    #     for ni in neighbors[n]:
                    #         if ni != out_of_bounds_entry:
                    #             if n not in neighbors[ni]:
                    #                 print("ugh")

    # clean up neighbor sets to remove any adjacent neighbors with the same sparse codes (avoid cotan issues)
    for node in neighbors:
        nbrs = neighbors[node]
        idx = 0
        while idx < len(nbrs) and len(nbrs) > 1:
            n = nbrs[idx]
            ni = nbrs[idx - 1]
            if n == out_of_bounds_entry:
                if ni == out_of_bounds_entry:
                    del nbrs[idx - 1]
                else:
                    idx += 1
            elif ni == out_of_bounds_entry:
                idx += 1
            else:
                n_val = img_alphas[:, n[0] * n_patch_per_dim + n[1]]
                ni_val = img_alphas[:, ni[0] * n_patch_per_dim + ni[1]]
                if np.all(n_val == ni_val):
                    del nbrs[idx - 1]
                else:
                    # only move forward once this neighbor + its previous neighbor in the list are different
                    idx += 1
        neighbors[node] = nbrs

    return valid_idx, neighbors

def prune_to_delaunay(valid_idx, neighbors, grid, n_patch_per_dim, img_alphas):
    """Given the vertex set and edge set of a triangle mesh (with potentially overlapping edges), return the vertex and edge set of the corresponding Delaunay triangulation."""
    
    # convert given connections into a triangulation without overlapping edges
    valid_idx, neighbors = prune_to_triangulation(valid_idx, neighbors, grid, n_patch_per_dim, img_alphas)


    # TODO(as) while not Delaunay, flip non-Delaunay edges


    return valid_idx, neighbors

def _get_all_neighbors(x, y, n_patch_per_dim):
    # Return all neighbors, including indices possibly outside the range of the image
    assert x >= 0 and x < n_patch_per_dim
    assert y >= 0 and y < n_patch_per_dim
    
    candidates = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    out = []
    for c in candidates:
        xo, yo = c
        if xo == 0 and yo == 0:
            continue
        x_off = x + xo
        y_off = y + yo
        out.append((x_off, y_off))
    return out

def _get_valid_neighbors(x, y, n_patch_per_dim):
    # Returns indices of all neighbors of the given index in the bounds of the image in counterclockwise order beginning with (-1, -1)
    out = []
    for c in _get_all_neighbors(x, y, n_patch_per_dim):
        x_off, y_off = c
        if x_off >= 0 and x_off < n_patch_per_dim and y_off >= 0 and y_off < n_patch_per_dim:
            out.append((x_off, y_off))
    return out

def _combine_neighbors(grid, img_alphas, i, j, n_patch_per_dim):
    # Use BFS until no more neighbors can be combined with the given (i, j) index
    val = img_alphas[:, i * n_patch_per_dim + j]
    q = [(i, j)]
    seen = set()
    while len(q) > 0:
        top = q.pop(0)
        if top not in seen and np.all(img_alphas[:, top[0] * n_patch_per_dim + top[1]] == val):
            assert grid[top] == top     # sanity check, if this is not the case then another search could have been expanded further
            grid[top] = (i, j)
            seen.add(top)

            # enqeue all neighbors of this node to be checked as well
            q.extend(_get_valid_neighbors(top[0], top[1], n_patch_per_dim))
    return grid

def _simplify_neighbor_list(nbrs):
    # Combine any adjacent duplicate neighbors
    idx = 0
    while idx < len(nbrs) and len(nbrs) > 1:
        n = nbrs[idx]
        ni = nbrs[idx - 1]
        if n == ni:     # all "neighbors" have been reduced to their connected component parents, can just check equality here
            del nbrs[idx - 1]
        else:           # only move forward once this neighbor + its previous neighbor in the list are different
            idx += 1
    return nbrs

def _follow_boundary(i, j, grid, n_patch_per_dim):
    # Walk the boundary of the connected component defined by grid[(x, y)] == grid[(i, j)]. Assumes (i, j) is the first instance of this connected component found
    # during a raster scan of the image so (i, j) cannot be an internal node of the connected component (it must have neighbors not in the connected component)
    # Generates the boundary walk in the clockwise direction
    boundary = []
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]         # clockwise
    cur_dir = 0
    current_pt = (i, j)
    start_out_dir = set()

    while True:
        boundary.append(current_pt)
        prev_pt = current_pt

        # check if neighbors of current point are part of the same connected component in clockwise order
        found = False
        for _ in range(len(directions)):
            next_pt = (current_pt[0] + directions[cur_dir][0], current_pt[1] + directions[cur_dir][1])
            if 0 <= next_pt[0] < n_patch_per_dim and 0 <= next_pt[1] < n_patch_per_dim and grid[next_pt] == grid[(i, j)]:
                current_pt = next_pt
                found = True
                break
            cur_dir = (cur_dir + 1) % len(directions)
        if not found:
            break       # dead end!
        
        if prev_pt == (i, j):
            if cur_dir in start_out_dir:
                # we've already taken this path out of the start node, done here
                boundary = boundary[:-1]        # remove duplicate start node
                break
            start_out_dir.add(cur_dir)

        # start searching for next boundary node at the element in the clockwise rotation that follows the node we came from
        cur_dir = (cur_dir + 5) % len(directions)
    return boundary

def _find_shared_start_end_seq(a, b):
    """Return the length of the common subsequence between the end of list "a" and the start of list "b" """
    # find last occurrence of the first element of "b" in "a". the start of the possible common subsequence
    first_el = b[0]
    if first_el not in a:
        return 0
    a_last_occ_idx = len(a) - 1 - a[::-1].index(first_el)

    # verify that this final chunk of N elements of "a" matches the first N elements of "b"
    a_chnk = a[a_last_occ_idx:]
    if a_chnk == b[:len(a_chnk)]:
        return len(a_chnk)
    else:
        return 0

def _convert_boundary_walk_to_neighbor_list(boundary_walk, grid, n_patch_per_dim):
    # Given a list of boundary vertices of a connected component, return an ordered list of neighbors of the connected component
    
    nbrs = []
    out_of_bounds_entry = (-1, -1)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]         # clockwise
    if len(boundary_walk) == 1:
        # single member connected component (no node collapsing)
        pt = boundary_walk[0]
        for d in directions:
            nb = (pt[0] + d[0], pt[1] + d[1])
            if 0 <= nb[0] < n_patch_per_dim and 0 <= nb[1] < n_patch_per_dim:
                nbrs.append(grid[nb])
            else:
                nbrs.append(out_of_bounds_entry)
        return _simplify_neighbor_list(nbrs)

    for idx in range(len(boundary_walk)):
        prev = boundary_walk[idx-1]     # when idx == 0, wraps around to end of the boundary walk. boundary walk is a cycle
        node = boundary_walk[idx]
        local_nbrs = []
        
        # starting with the neighbor after the previous node in the boundary walk, iterate through all neighbors of node until a member of the connected component is found
        prev_to_node_dir = (prev[0] - node[0], prev[1] - node[1])
        dir_idx = directions.index(prev_to_node_dir) + 1
        for i in range(len(directions)):
            offset = directions[(dir_idx + i) % len(directions)]
            nbr_grid_idx = (node[0] + offset[0], node[1] + offset[1])
            if 0 <= nbr_grid_idx[0] < n_patch_per_dim and 0 <= nbr_grid_idx[1] < n_patch_per_dim:
                if grid[nbr_grid_idx] == grid[node]:
                    break
                local_nbrs.append(grid[nbr_grid_idx])
            else:    
                local_nbrs.append(out_of_bounds_entry)

        if idx == 0:
            nbrs = local_nbrs
        else:
            # appending local to global, remove any duplicate neighbors from the start of local first
            local_start_global_end_overlap = _find_shared_start_end_seq(nbrs, local_nbrs)
            local_nbrs = local_nbrs[local_start_global_end_overlap:]
            nbrs.extend(local_nbrs)

    # remove any overlap between start and end of the neighbor list
    if len(boundary_walk) > 1:
        start_end_overlap = _find_shared_start_end_seq(nbrs, nbrs) 
        nbrs = nbrs[:(-1*start_end_overlap)]

    return _simplify_neighbor_list(nbrs)

def _get_neighbor_set(grid, n_patch_per_dim):
    """
    Grid has been populated with connected component information. Extract neighbor sets by walking the boundary of each connected component and recording the boundary walk order.
    """
    neighbors = {}
    for i in range(n_patch_per_dim):
        for j in range(n_patch_per_dim):
            parent = grid[(i, j)]
            if parent not in neighbors:
                # collect ordered boundary walks for this connected component
                boundary_walk = _follow_boundary(i, j, grid, n_patch_per_dim)
                
                # generate an ordered neighbor list from the boundary walk
                neighbors[parent] = _convert_boundary_walk_to_neighbor_list(boundary_walk, grid, n_patch_per_dim)
    return list(neighbors.keys()), neighbors

def collapse_neighbors(n_patch_per_dim, img_alphas, img_idx):
    """
    Combine any adjacent points with the same sparse encoding such that the resulting combined point has the appropriately ordered neighbor set of both.
    Assume a context size of 1.
    
    0 1 2 3             0 1 2 3 
    4 * * 5     -->     4  *  5
    6 7 8 9             6 7 8 9

    """

    # Create a full-size grid of "pointers" to the node that now represents that space
    #   For each index, if the pointer is equal to its index in the grid try to expand it by searching over all neighboring indices (context range of 1) to see if they have the same sparse code.
    #   (sanity check: neighbor point should ALWAYS equal their grid index otherwise another index could have been expanded further). If sparse code is the same, make neighbor pointer point to 
    #   current index. Use BFS/DFS until no more neighbors can be converted, then continue top-level iteration.
    grid = {(i, j):(i, j) for i in range(n_patch_per_dim) for j in range(n_patch_per_dim)}
    for i in range(n_patch_per_dim):
        for j in range(n_patch_per_dim):
            if grid[(i, j)] == (i, j):
                grid = _combine_neighbors(grid, img_alphas, i, j, n_patch_per_dim)

    # Final pass to determine neighbors + their order
    #   Given grid of pointers, for each index that points to itself collect an ordered list of unqiue neighbors (in a counterclockwise fashion starting with the top left). This should result 
    #   in a final list of indices that point to themselves as well as a dictionary that maps each of these indices to the ordered list of unique neighbors
    indices, neighbors = _get_neighbor_set(grid, n_patch_per_dim)

    # Prune any non-symmetric neighbor relations (only happens when a pixel with a different value 
    # is inside of a connected component, boundary following and neighbors logic do not handle holes 
    # in the connected components)
    tot_rm = 0
    tot = 0
    for n in neighbors:
        rm = []
        for ni in neighbors[n]:
            if ni != (-1, -1) and n not in neighbors[ni]:
                rm.append(ni)
        tot_rm += len(rm)
        tot += len(neighbors[n])
        for r in rm:
            neighbors[n].remove(r)
    if tot_rm > 0:
        print(f"({img_idx}): Pruned {tot_rm} / {tot} assymetric neighbor relations.")

    return indices, neighbors, grid

def calc_cotan(a, b, c):
    """
    Calculate the cotangent of the angle formed by vectors AB and BC. The cotangent of the angle at B.
    Normally: cotan = dot(x, y) / ||cross(x, y)||
    But R^n has no (natural) cross product, so instead get cos from dot product and convert this to cot using trig id:
        cot = +/- cos / (\sqrt{1 - cos^2})
    Given triangle context, we can safely select the positive angle. Rewrite as:
        cot = dot(x, y) / \sqrt{||x||^2 ||y||^2 - dot(x, y)^2}
    """
    x = (a - b).squeeze()       # vector from B to A
    y = (c - b).squeeze()       # vector from B to C
    dot = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    diff = norm_x**2 * norm_y**2 - dot**2
    if diff <= 1e-2:
        # a >0 to handle FP error and avoid returning very large cotangent values
        # in essence, we want to avoid comparing points that already have the same embedding (zero area triangles)
        print(f"\tinvalid cot: {diff}, {dot}, {norm_x}, {norm_y}")
        return None, False
    cot = dot / np.sqrt(diff)
    return cot, True

def sparse_cotan_laplacian(img_alphas, img_idx, n_patch_per_img, zero_codes, area_norm):
    """Return the cotan Laplacian matrix for the given image as a sparse matrix."""

    n_patch_per_dim = int(np.sqrt(n_patch_per_img))      # assume square images
    assert n_patch_per_dim**2 == n_patch_per_img

    # prefetch alphas codes for all patches in this image. faster to index into this image-specific set of sparse codes
    img_alphas = img_alphas.todense()

    # "collapse" neighboring points with the same sparse code to remove zero area triangles
    valid_idx, neighbors, grid = collapse_neighbors(n_patch_per_dim, img_alphas, img_idx)

    # convert current neighbor sets to Delaunay triangulation
    # valid_idx, neighbors = prune_to_delaunay(valid_idx, neighbors, grid, n_patch_per_dim, img_alphas)

    lap = np.zeros(shape=(n_patch_per_img, n_patch_per_img), dtype=np.float32)
    skipped_invalid = 0
    out_of_bounds_entry = (-1, -1)
    for idx in valid_idx:            
        x, y = idx
        center_idx = x * n_patch_per_dim + y

        if len(neighbors[idx]) < 2:
            skipped_invalid += 1
            lap[center_idx, :] = 0
            continue
        
        if _neighborhood_has_zero_code(x, y, zero_codes, n_patch_per_dim):
            # skip if this patch or any of its neighbors have a zero code
            continue

        # pre-fetch alphas values for each member of the neighorhood (remove duplicate column slicing)
        center_val = img_alphas[:, center_idx]
        alphas_cache = {}
        for n in neighbors[idx]:
            if n != out_of_bounds_entry:
                alphas_cache[n] = img_alphas[:, n[0] * n_patch_per_dim + n[1]]

        # calculate per-neighbor weight
        n_invalid_neighbors = 0
        for i, neighbor in enumerate(neighbors[idx]):
            if neighbor == out_of_bounds_entry:
                continue
            
            n_idx = neighbor[0] * n_patch_per_dim + neighbor[1]
            n_val = alphas_cache[neighbor]

            valid = True
            nxt_idx = neighbors[idx][(i + 1) % len(neighbors[idx])]
            prev_idx = neighbors[idx][i - 1]
            tot = 0
            if nxt_idx != out_of_bounds_entry:
                nxt_val = alphas_cache[nxt_idx]
                cot_beta, valid = calc_cotan(center_val, nxt_val, n_val)
                if valid:
                    tot += cot_beta
            if valid and prev_idx != out_of_bounds_entry:
                prev_val = alphas_cache[prev_idx]        # python negative indexing should handle wrap around in this case
                cot_alpha, valid = calc_cotan(center_val, prev_val, n_val)
                if valid:
                    tot += cot_alpha
            if not valid:
                n_invalid_neighbors += 1
                continue

            # interior edge: (cot_alpha + cot_beta) / 2
            # boundary edge: cot / 2
            # https://arxiv.org/pdf/math/0503219 
            lap[center_idx, n_idx] = tot / 2

        if (len(neighbors[idx]) - n_invalid_neighbors) < 2:
            skipped_invalid += 1
            lap[center_idx, :] = 0
            continue

        lap[center_idx, center_idx] = -1 * np.sum(lap[center_idx, :])


        if area_norm:

            # TODO(as) need to fix this now that neighbors are handled differently. minor fix just not needed right now

            # calculate vertex area as 1/3rd of the area of all triangles that include it vertex
            node_weight = 0
            for idx in range(len(offset_order)):
                cur_val = alphas_cache[offset_order[idx]]
                prior_val = alphas_cache[offset_order[idx-1]]       # python list indexing properly handles negative indexing (wrap around)

                # Calculate using SAS triangle and some algebra
                u = prior_val - center_val
                v = cur_val - center_val
                u_norm = np.linalg.norm(u)
                v_norm = np.linalg.norm(v)
                if u_norm == 0 or v_norm == 0:
                    continue

                dot = np.dot(u, v)
                diff = (u_norm**2 * v_norm**2) - dot**2
                if diff <= 0:
                    # may be slightly less than zero due to numerical instability
                    continue

                node_weight += np.sqrt(diff) / 2
            
            # "vertex" area is 1/3 of area of each triangle that contains it
            node_weight /= 3
        else:
            # uniform weight normalization
            node_weight = len([i for i in neighbors[idx] if i != out_of_bounds_entry])

        # scale by center node weight to get full laplacian
        assert node_weight > 0
        lap[center_idx, :] /= node_weight 

    if skipped_invalid > 0:
        print(f"Image {img_idx}: skipped {skipped_invalid} / {n_patch_per_img} patches.")
    
    # sanity check that rows must sum to zero
    # sanity = np.sum(lap, axis=1)
    # assert np.abs(sanity.max()) < 1e-5 and np.abs(sanity.min()) < 1e-5

    return lap


def _neighborhood_has_zero_code(x, y, zero_codes, n_patch_per_dim):
    # Return true if given patch or its immediate neighbors have zero codes (does not validate neighbors are in bounds of image)
    for x_off in [-1, 0, 1]:
        for y_off in [-1, 0, 1]:
            if (x+x_off) * n_patch_per_dim + (y+y_off) in zero_codes:
                return True
    return False
    


