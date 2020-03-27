# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2019, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# a BSD-style license.  See the LICENSE file in the package base
# directory for details.


from typing import List

from graph_tool import Vertex

import PETGraph
from pattern_detectors.PatternInfo import PatternInfo
from utils import find_subnodes, depends, calculate_workload, \
    total_instructions_count, classify_task_vars

__forks = set()
__workloadThreshold = 10000
__minParallelism = 3


class Task(object):
    """This class represents task in task parallelism pattern
    """
    nodes: List[Vertex]
    child_tasks: List['Task']
    start_line: str
    end_line: str

    def __init__(self, pet: PETGraph, node: Vertex):
        self.node_id = pet.graph.vp.id[node]
        self.nodes = [node]
        self.start_line = pet.graph.vp.startsAtLine[node]
        self.end_line = pet.graph.vp.endsAtLine[node]
        self.mw_type = pet.graph.vp.mwType[node]
        self.instruction_count = total_instructions_count(pet, node)
        self.workload = calculate_workload(pet, node)
        self.child_tasks = []

    def aggregate(self, other: 'Task'):
        """Aggregates given task into current task

        :param other: task to aggregate
        """
        self.nodes.extend(other.nodes)
        self.end_line = other.end_line
        self.workload += other.workload
        self.instruction_count += other.instruction_count
        self.mw_type = 'BARRIER_WORKER' if other.mw_type == 'BARRIER_WORKER' else 'WORKER'


def __merge_tasks(pet: PETGraph, task: Task):
    """Merges the tasks into having required workload.

    :param pet: PET graph
    :param task: task node
    """
    for i in range(len(task.child_tasks)):
        child_task: Task = task.child_tasks[i]
        if child_task.workload < __workloadThreshold:  # todo child child_tasks?
            if i > 0:
                pred: Task = task.child_tasks[i - 1]
                if __neighbours(pred, child_task):
                    pred.aggregate(child_task)
                    pred.child_tasks.remove(child_task)
                    __merge_tasks(pet, task)
                    return
            if i + 1 < len(task.child_tasks) - 1:  # todo off by one?, elif?
                succ: Task = task.child_tasks[i + 1]
                if __neighbours(child_task, succ):
                    child_task.aggregate(succ)  # todo odd aggregation in c++
                    task.child_tasks.remove(succ)
                    __merge_tasks(pet, task)
                    return
            task.child_tasks.remove(child_task)
            __merge_tasks(pet, task)
            return

    if task.child_tasks and len(task.child_tasks) < __minParallelism:
        max_workload_task = max(task.child_tasks, key=lambda t: t.workload)
        task.child_tasks.extend(max_workload_task.child_tasks)
        task.child_tasks.remove(max_workload_task)
        __merge_tasks(pet, task)
        return

    for child in task.child_tasks:
        if pet.graph.vp.type[child.nodes[0]] == 'loop':
            pass  # todo add loops?


def __neighbours(first: Task, second: Task):
    """Checks if second task immediately follows first task

    :param first: predecessor task
    :param second: successor task
    :return: true if second task immediately follows first task
    """
    fel = int(first.end_line.split(':')[1])
    ssl = int(second.start_line.split(':')[1])
    return fel == ssl or fel + 1 == ssl or fel + 2 == ssl


class TaskParallelismInfo(PatternInfo):
    """Class, that contains task parallelism detection result
    """

    def __init__(self, pet: PETGraph, node: Vertex, pragma, pragma_line, first_private, private, shared):
        """
        :param pet: PET graph
        :param node: node, where task parallelism was detected
        :param pragma: pragma to be used (task / taskwait)
        :param pragma_line: line prior to which the pragma shall be inserted
        :param first_private: list of varNames
        :param private: list of varNames
        :param shared: list of varNames
        """
        PatternInfo.__init__(self, pet, node)
        self.pragma = pragma
        self.pragma_line = pragma_line
        self.first_private = first_private
        self.private = private
        self.shared = shared

    def __str__(self):
        return f'Task parallelism at CU: {self.node_id}\n' \
               f'CU Start line: {self.start_line}\n' \
               f'CU End line: {self.end_line}\n' \
               f'pragma prior to line: {self.pragma_line}\n' \
               f'pragma: "#pragma omp {" ".join(self.pragma)}"\n' \
               f'first_private: {" ".join(self.first_private)}\n' \
               f'private: {" ".join(self.private)}\n' \
               f'shared: {" ".join(self.shared)}'


def run_detection(pet: PETGraph) -> List[TaskParallelismInfo]:
    """Computes the Task Parallelism Pattern for a node:
    (Automatic Parallel Pattern Detection in the Algorithm Structure Design Space p.46)
    1.) first merge all children of the node -> all children nodes get the dependencies
        of their children nodes and the list of the children nodes (saved in node.childrenNodes)
    2.) To detect Task Parallelism, we use Breadth First Search (BFS)
        a.) the hotspot becomes a fork
        b.) all child nodes become first worker if they are not marked as worker before
        c.) if a child has dependence to more than one parent node, it will be marked as barrier
    3.) if two barriers can run in parallel they are marked as barrierWorkers.
        Two barriers can run in parallel if there is not a directed path from one to the other

        :param pet: PET graph
        :return: List of detected pattern info
    """
    result = []

    for node in pet.graph.vertices():
        if pet.graph.vp.type[node] == 'dummy':
            continue
        if find_subnodes(pet, node, 'child'):
            # print(graph.vp.id[node])
            __detect_mw_types(pet, node)

        if pet.graph.vp.mwType[node] == 'NONE':
            pet.graph.vp.mwType[node] = 'ROOT'

    __forks.clear()
    __create_task_tree(pet, pet.main)

    # ct = [graph.vp.id[v] for v in pet.graph.vp.childrenTasks[main_node]]
    # ctt = [graph.vp.id[v] for v in forks]
    fs = [f for f in __forks if f.node_id == '130:0']
    for fork in fs:
        # todo __merge_tasks(graph, fork)
        if fork.child_tasks:
            result.append(TaskParallelismInfo(pet, fork.nodes[0], [], [], [], [], []))

    result = result + __detect_task_suggestions(pet)

    return result


def __detect_task_suggestions(pet: PETGraph):
    """creates task parallelism suggestions and returns them as a list of
    TaskParallelismInfo objects.
    Currently relies on previous processing steps and suggests WORKER CUs
    as Tasks and BARRIER/BARRIER_WORKER as Taskwaits.

    :param pet: PET graph
    :return List[TaskParallelismInfo]
    """
    # suggestions contains a map from LID to a set of suggestions. This is required to
    # detect multiple suggestions for a single line of source code.
    suggestions = dict()  # LID -> set<list<set<string>>>
    # list[0] -> task / taskwait
    # list[1] -> vertex
    # list[2] -> pragma line number
    # list[3] -> first_private Clause
    # list[4] -> private clause
    # list[5] -> shared clause

    # get a list of cus classified as WORKER
    worker_cus = []
    barrier_cus = []
    barrier_worker_cus = []

    for v in pet.graph.vertices():
        if pet.graph.vp.mwType[v] == "WORKER":
            worker_cus.append(v)
        if pet.graph.vp.mwType[v] == "BARRIER":
            barrier_cus.append(v)
        if pet.graph.vp.mwType[v] == "BARRIER_WORKER":
            barrier_worker_cus.append(v)

    # SUGGEST TASKWAIT
    for v in barrier_cus + barrier_worker_cus:
        tmp_suggestion = [["taskwait"], v, pet.graph.vp.startsAtLine[v], [], [], []]
        if pet.graph.vp.startsAtLine[v] not in suggestions:
            # no entry for source code line contained in suggestions
            tmp_set = []
            suggestions[pet.graph.vp.startsAtLine[v]] = tmp_set
            suggestions[pet.graph.vp.startsAtLine[v]].append(tmp_suggestion)
        else:
            # entry for source code line already contained in suggestions
            suggestions[pet.graph.vp.startsAtLine[v]].append(tmp_suggestion)

    # SUGGEST TASKS
    for vx in pet.graph.vertices():
        # iterate over all entries in recursiveFunctionCalls
        # in order to find task suggestions
        for i in range(0, len(pet.graph.vp.recursiveFunctionCalls[vx])):
            function_call_string = pet.graph.vp.recursiveFunctionCalls[vx][i]
            if not type(function_call_string) == str:
                continue
            contained_in = __recursive_function_call_contained_in_worker_cu(
                pet, function_call_string, worker_cus)
            if contained_in is not None:
                current_suggestions = [[], None, None, [], [], []]
                # recursive Function call contained in worker cu
                # -> issue task suggestion
                pragma_line = function_call_string[
                              function_call_string.index(":") + 1:]
                pragma_line = pragma_line.replace(",", "").replace(" ", "")

                # only include cu and func nodes
                if not ('func' in pet.graph.vp.type[contained_in] or
                        "cu" in pet.graph.vp.type[contained_in]):
                    continue

                if pet.graph.vp.mwType[contained_in] == "WORKER":
                    # suggest task
                    fpriv, priv, shared, in_dep, out_dep, in_out_dep, red = \
                        classify_task_vars(pet, contained_in, "", [], [])
                    current_suggestions[0].append("task")
                    current_suggestions[1] = vx
                    current_suggestions[2] = pragma_line
                    for var_id in fpriv:
                        current_suggestions[3].append(var_id.name)
                    for var_id in priv:
                        current_suggestions[4].append(var_id.name)
                    for var_id in shared:
                        current_suggestions[5].append(var_id.name)

                # insert current_suggestions into suggestions
                # check, if current_suggestions contains an element
                if len(current_suggestions[0]) >= 1:
                    # current_suggestions contains something
                    if pragma_line not in suggestions:
                        # LID not contained in suggestions
                        tmp_set = []
                        suggestions[pragma_line] = tmp_set
                        suggestions[pragma_line].append(current_suggestions)
                    else:
                        # LID already contained in suggestions
                        suggestions[pragma_line].append(current_suggestions)
    # end of for loop

    # construct return value (list of TaskParallelismInfo)
    result = []
    for key in suggestions:
        for single_suggestion in suggestions[key]:
            pragma = single_suggestion[0]
            node = single_suggestion[1]
            pragma_line = single_suggestion[2]
            first_private = single_suggestion[3]
            private = single_suggestion[4]
            shared = single_suggestion[5]
            result.append(TaskParallelismInfo(pet, node, pragma, pragma_line,
                                              first_private, private, shared))
    return result


def __recursive_function_call_contained_in_worker_cu(pet: PETGraph,
                                                     function_call_string: str,
                                                     worker_cus: [Vertex]):
    """check if submitted function call is contained in at least one WORKER cu.
    Returns the vertex identifier of the containing cu.
    If no cu contains the function call, None is returned.
    Note: The Strings stored in recursiveFunctionCalls might contain multiple function calls at once.
          in order to apply this function correctly, make sure to split Strings in advance and supply
          one call at a time.
    :param pet: PET graph
    :param function_call_string: String representation of the recursive function call to be checked
            Ex.: fib 7:35,  (might contain ,)
    :param worker_cus: List of vertices
    """
    # remove , and whitespaces at start / end
    function_call_string = function_call_string.replace(",", "")
    while function_call_string.startswith(" "):
        function_call_string = function_call_string[1:]
    while function_call_string.endswith(" "):
        function_call_string = function_call_string[:-1]
    # function_call_string looks now like like: 'fib 7:52'

    # split String into function_name. file_id and line_number
    function_name = function_call_string[0:function_call_string.index(" ")]
    file_id = function_call_string[
              function_call_string.index(" ") + 1:
              function_call_string.index(":")]
    line_number = function_call_string[function_call_string.index(":") + 1:]

    # iterate over worker_cus
    for cur_w in worker_cus:
        cur_w_starts_at_line = pet.graph.vp.startsAtLine[cur_w]
        cur_w_ends_at_line = pet.graph.vp.endsAtLine[cur_w]
        cur_w_file_id = cur_w_starts_at_line[:cur_w_starts_at_line.index(":")]
        # check if file_id is equal
        if file_id == cur_w_file_id:
            # trim to line numbers only
            cur_w_starts_at_line = cur_w_starts_at_line[
                                   cur_w_starts_at_line.index(":") + 1:]
            cur_w_ends_at_line = cur_w_ends_at_line[
                                 cur_w_ends_at_line.index(":") + 1:]
            # check if line_number is contained
            if int(cur_w_starts_at_line) <= int(line_number) <= int(cur_w_ends_at_line):
                return cur_w
    return None


def __detect_mw_types(pet: PETGraph, main_node: Vertex):
    """The mainNode we want to compute the Task Parallelism Pattern for it
    use Breadth First Search (BFS) to detect all barriers and workers.
    1.) all child nodes become first worker if they are not marked as worker before
    2.) if a child has dependence to more than one parent node, it will be marked as barrier
    Returns list of BARRIER_WORKER pairs 2
    :param pet: PET graph
    :param main_node: root node
    """

    # first insert all the direct children of main node in a queue to use it for the BFS
    for node in find_subnodes(pet, main_node, 'child'):
        # a child node can be set to NONE or ROOT due a former detectMWNode call where it was the mainNode
        if pet.graph.vp.mwType[node] == 'NONE' or pet.graph.vp.mwType[node] == 'ROOT':
            pet.graph.vp.mwType[node] = 'FORK'

        # while using the node as the base child, we copy all the other children in a copy vector.
        # we do that because it could be possible that two children of the current node (two dependency)
        # point to two different children of another child node which results that the child node becomes BARRIER
        # instead of WORKER
        # so we copy the whole other children in another vector and when one of the children of the current node
        # does point to the other child node, we just adjust mwType and then we remove the node from the vector
        # Thus we prevent changing to BARRIER due of two dependencies pointing to two different children of
        # the other node

        # create the copy vector so that it only contains the other nodes
        other_nodes = find_subnodes(pet, main_node, 'child')
        other_nodes.remove(node)

        for other_node in other_nodes:
            if depends(pet, other_node, node):
                # print("\t" + pet.graph.vp.id[node] + "<--" + pet.graph.vp.id[other_node])
                if pet.graph.vp.mwType[other_node] == 'WORKER':
                    pet.graph.vp.mwType[other_node] = 'BARRIER'
                else:
                    pet.graph.vp.mwType[other_node] = 'WORKER'

                    # check if other_node has > 1 RAW dependencies to node
                    # -> not detected in previous step, since other_node is only
                    #    dependent of a single CU
                    raw_targets = []
                    for e in other_node.out_edges():
                        if e.target() == node:
                            if pet.graph.ep.dtype[e] == 'RAW':
                                raw_targets.append(pet.graph.vp.id[e.target()])
                    # remove entries which occur less than two times
                    raw_targets = [t for t in raw_targets if raw_targets.count(t) > 1]
                    # remove duplicates from list
                    raw_targets = list(set(raw_targets))
                    # if elements remaining, mark other_node as BARRIER
                    if len(raw_targets) > 0:
                        pet.graph.vp.mwType[other_node] = 'BARRIER'

    pairs = []
    # check for Barrier Worker pairs
    # if two barriers don't have any dependency to each other then they create a barrierWorker pair
    # so check every barrier pair that they don't have a dependency to each other -> barrierWorker
    direct_subnodes = find_subnodes(pet, main_node, 'child')
    for n1 in direct_subnodes:
        if pet.graph.vp.mwType[n1] == 'BARRIER':
            for n2 in direct_subnodes:
                if pet.graph.vp.mwType[n2] == 'BARRIER' and n1 != n2:
                    if n2 in [e.target() for e in n1.out_edges()] or n2 in [e.source() for e in n1.in_edges()]:
                        break
                    # so these two nodes are BarrierWorker, because there is no dependency between them
                    pairs.append((n1, n2))
                    pet.graph.vp.mwType[n1] = 'BARRIER_WORKER'
                    pet.graph.vp.mwType[n2] = 'BARRIER_WORKER'

    # return pairs


def __create_task_tree(pet: PETGraph, root: Vertex):
    """generates task tree data from root node

    :param pet: PET graph
    :param root: root node
    """
    root_task = Task(pet, root)
    __forks.add(root_task)
    __create_task_tree_helper(pet, root, root_task, [])


def __create_task_tree_helper(pet: PETGraph, current: Vertex, root: Task, visited_func: List[Vertex]):
    """generates task tree data recursively

    :param pet: PET graph
    :param current: current vertex to process
    :param root: root task for subtree
    :param visited_func: visited function nodes
    """
    if pet.graph.vp.type[current] == 'func':
        if current in visited_func:
            return
        else:
            visited_func.append(current)

    for child in find_subnodes(pet, current, 'child'):
        mw_type = pet.graph.vp.mwType[child]

        if mw_type in ['BARRIER', 'BARRIER_WORKER', 'WORKER']:
            task = Task(pet, child)
            root.child_tasks.append(task)
            __create_task_tree_helper(pet, child, task, visited_func)
        elif mw_type == 'FORK' and not pet.graph.vp.startsAtLine[child].endswith('16383'):
            task = Task(pet, child)
            __forks.add(task)
            __create_task_tree_helper(pet, child, task, visited_func)
        else:
            __create_task_tree_helper(pet, child, root, visited_func)
