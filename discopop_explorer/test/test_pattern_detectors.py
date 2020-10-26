import unittest

import networkx as nx  # type:ignore

import discopop_explorer.pattern_detectors.reduction_detector as reduction_detector
from discopop_explorer.PETGraphX import CUNode, Dependency, EdgeType, NodeType, PETGraphX
from discopop_explorer.variable import Variable


class ReductionDetectorTest(unittest.TestCase):
    def test_reduction_detection(self):
        """Simple loop reducing on one variable"""
        g = nx.MultiDiGraph()
        loop_node = CUNode.from_kwargs(
            node_id="0:0",
            type=NodeType.LOOP,
            name="main",
            source_file=0,
            start_line=0,
            end_line=0,
            loop_iterations=1,
        )
        g.add_node(loop_node.id, data=loop_node)
        var_node = CUNode.from_kwargs(
            node_id="0:1", type=NodeType.CU, name="var", local_vars=[Variable("int", "x")]
        )
        g.add_node(var_node.id, data=var_node)
        g.add_edge(loop_node.id, var_node.id, data=Dependency(EdgeType.CHILD))
        reduction_vars = [
            {
                "loop_line": f"{loop_node.source_file}:{loop_node.start_line}",
                "name": var_node.local_vars[0].name,
            }
        ]
        with self.subTest("Reduction Pattern"):
            patterns = reduction_detector.run_detection(PETGraphX(g, reduction_vars, {}))
            self.assertListEqual([pattern.node_id for pattern in patterns], [loop_node.id])
        with self.subTest("Reduction Pattern w/out loop iterations"):
            loop_node.loop_iterations = 0
            patterns = reduction_detector.run_detection(PETGraphX(g, reduction_vars, {}))
            self.assertListEqual(patterns, [])
            loop_node.loop_iterations = 1
        with self.subTest("Reduction pattern w/out reduction vars"):
            patterns = reduction_detector.run_detection(PETGraphX(g, [], {}))
            self.assertListEqual(patterns, [])
        with self.subTest("Reduction pattern with wrong reduction var"):
            reduction_vars[0]["name"] += "_"
            patterns = reduction_detector.run_detection(PETGraphX(g, reduction_vars, {}))
            self.assertListEqual(patterns, [])
