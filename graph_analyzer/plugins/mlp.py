from copy import deepcopy
from typing import List, Set

from PETGraphX import PETGraphX, NodeType, CUNode, EdgeType, DepType, Dependency
from utils import correlation_coefficient


def run_before(pet: PETGraphX):
    return pet


def run_after(pet: PETGraphX):
    for node in pet.all_nodes(NodeType.LOOP):
        mlp_loops = get_mlp_loops(pet, node)
        if len(mlp_loops) > 1:
            print(" ".join([l.id for l in mlp_loops]))
            check_mlp(pet, mlp_loops)

    return pet


def get_mlp_loops(pet: PETGraphX, root: CUNode) -> List[CUNode]:
    """Finds a sequence of loops that follow each other immediately

    :param pet: CU graph
    :param root: first mlp loop
    :return:
    """
    res = [root]
    children = pet.direct_children(root)
    next = None

    # find for end via condition node in loop children
    # (end block is not in loop children)
    for c in children:
        if "for.cond" in c.basic_block_id:
            for c2 in pet.successors_of(c):
                if "for.end" in c2.basic_block_id:
                    successors = pet.successors_of(c2)
                    next = successors[0] if successors else None
            break

    if next is not None and "for.cond" in next.basic_block_id:
        res.extend(get_mlp_loops(pet, pet.parent_of(next)))

    return res


def check_mlp(pet: PETGraphX, root_nodes: List[CUNode]):
    """Checks, whether the nodes represent multi-loop pipeline
    Tries to optimize dependencies for multi-loop pipeline detection
    1. Deletes independent lines, that do not contribute to the pipeline
    2. Deletes similar CU (that have same dependencies), as those can be one step in the pipeline

    :param pet: PET graph
    :param root_nodes: current node
    :return: Pipeline scalar value
    """
    children_start_lines = []
    loop_subnodes = []

    for l in root_nodes:
        children_start_lines.extend([v.start_position() for v in pet.subtree_of_type(l, NodeType.LOOP)])

        loop_subnodes.extend([pet.node_at(t) for s, t, d in pet.out_edges(l.id, EdgeType.CHILD)
                              if is_pipeline_subnode(l, pet.node_at(t), children_start_lines)])

    if len(loop_subnodes) < 2:
        return

    matrix = get_matrix(pet, root_nodes, loop_subnodes)
    initial_matrix = deepcopy(matrix)
    initial_coef = get_correlation_coefficient(matrix)

    print(f"matrix at {root_nodes[0].start_position()}: {initial_coef}")
    for i in range(0, len(initial_matrix)):
        print(" ".join([str(x) for x in initial_matrix[i]]))

    independent_cus = get_independent_lines(matrix)
    delete_lines(matrix, loop_subnodes, independent_cus)

    mergeable_cus = get_mergeable_nodes(matrix)
    delete_lines(matrix, loop_subnodes, mergeable_cus)

    new_coef = get_correlation_coefficient(matrix)
    print(f"final coef: {new_coef}")
    for i in range(0, len(matrix)):
        print(" ".join([str(x) for x in matrix[i]]))


def delete_lines(matrix, loop_nodes, lines):
    if lines:
        lines.sort(reverse=True)
        for i in range(0, len(lines)):
            del loop_nodes[lines[i]]
            del matrix[lines[i]]
            for j in range(0, len(matrix)):
                del matrix[j][lines[i]]


def get_independent_lines(matrix):
    res = []
    for i in range(0, len(matrix)):
        indep = True
        for j in range(0, len(matrix)):
            if i != j and (matrix[i][j] != 0 or matrix[j][i] != 0):
                indep = False
        if indep:
            res.append(i)
    return res


def get_mergeable_nodes(matrix):
    res = []
    for i in reversed(range(1, len(matrix))):
        if matrix[i] == matrix[i - 1]:
            same = True
            for j in range(1, len(matrix)):
                if matrix[j][i] != matrix[j][i - 1]:
                    same = False
            if same:
                res.append(i)
    return res


def get_matrix(pet, root, loop_subnodes):
    res = []
    for i in range(0, len(loop_subnodes)):
        res.append([])
        for j in range(0, len(loop_subnodes)):
            res[i].append(int(depends_ignore_readonly(pet, loop_subnodes[i], loop_subnodes[j], root)))
    return res


def depends_ignore_readonly(pet, source: CUNode, target: CUNode, root_loop: List[CUNode]) -> bool:
    children = pet.subtree_of_type(target, NodeType.CU)
    # TODO children.append(target)

    for dep in get_all_dependencies(pet, source, root_loop):
        if dep in children:
            return True
    return False


def get_all_dependencies(pet, node: CUNode, root_loops: List[CUNode]) -> Set[CUNode]:
    """Returns all data dependencies of the node and it's children
    This method ignores loop index and read only variables

    :param pet: CU graph
    :param node: node
    :param root_loops: root loops
    :return: list of all RAW dependencies of the node
    """
    dep_set = set()
    children = pet.subtree_of_type(node, NodeType.CU)

    subtree = []
    loops_start_lines = []
    for l in root_loops:
        loops_start_lines.extend([v.start_position() for v in pet.subtree_of_type(l, NodeType.LOOP)])
        subtree.extend(pet.subtree_of_type(l, NodeType.CU))

    for v in children:
        for t, d in [(t, d) for s, t, d in pet.out_edges(v.id, EdgeType.DATA) if d.dtype == DepType.RAW]:
            if (pet.is_loop_index(d.var_name, loops_start_lines, subtree)
                    or is_readonly_inside_loop_body(pet, d, root_loops)):
                continue
            dep_set.add(pet.node_at(t))

    return dep_set


def is_readonly_inside_loop_body(pet, dep: Dependency, root_loops: List[CUNode]) -> bool:
    """Checks, whether a variable is read-only in loop body

    :param pet: CU graph
    :param dep: dependency variable
    :param root_loops: root loops
    :return: true if variable is read-only in loop body
    """
    # TODO pass as param?
    loops_start_lines = []
    children = []

    for l in root_loops:
        loops_start_lines.extend([v.start_position() for v in pet.subtree_of_type(l, NodeType.LOOP)])
        children.extend(pet.subtree_of_type(l, NodeType.CU))

    for v in children:
        for t, d in [(t, d) for s, t, d in pet.out_edges(v.id, EdgeType.DATA)
                     if d.dtype == DepType.WAR or d.dtype == DepType.WAW]:
            # If there is a waw dependency for var, then var is written in loop
            # (sink is always inside loop for waw/war)
            if (dep.var_name == d.var_name
                    and not (d.sink in loops_start_lines)):
                return False
        for t, d in [(t, d) for s, t, d in pet.in_edges(v.id, EdgeType.DATA)
                     if d.dtype == DepType.RAW]:
            # If there is a reverse raw dependency for var, then var is written in loop
            # (source is always inside loop for reverse raw)
            if (dep.var_name == d.var_name
                    and not (d.source in loops_start_lines)):
                return False
    return True


def get_correlation_coefficient(matrix):
    graph_vector = []
    for i in range(0, len(matrix) - 1):
        graph_vector.append(matrix[i + 1][i])

    pipeline_vector = []
    for i in range(0, len(matrix) - 1):
        pipeline_vector.append(1)

    min_weight = 1
    for i in range(0, len(matrix) - 1):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] == 1:
                node_weight = 1 - (j - i) / (len(matrix) - 1)
                if min_weight > node_weight > 0:
                    min_weight = node_weight

    if min_weight == 1:
        graph_vector.append(0)
        pipeline_vector.append(0)
    else:
        graph_vector.append(1)
        pipeline_vector.append(min_weight)
    return round(correlation_coefficient(graph_vector, pipeline_vector), 2)


def is_pipeline_subnode(root: CUNode, current: CUNode, children_start_lines: List[str]) -> bool:
    """Checks if node is a valid subnode for pipeline

    :param root: root node
    :param current: current node
    :param children_start_lines: start lines of children loops
    :return: true if valid
    """

    # TODO check basic block id?
    r_start = root.start_position()
    r_end = root.end_position()
    c_start = current.start_position()
    c_end = current.end_position()
    return not (c_start == r_start and c_end == r_start
                or c_start == r_end and c_end == r_end
                or c_start == c_end and c_start in children_start_lines)
