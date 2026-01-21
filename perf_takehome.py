"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.enable_vdebug = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized across the batch dimension with a scalar tail.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector constants and broadcasted parameters
        zero_v = self.alloc_scratch("zero_v", VLEN)
        one_v = self.alloc_scratch("one_v", VLEN)
        two_v = self.alloc_scratch("two_v", VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)
        forest_base_v = self.alloc_scratch("forest_base_v", VLEN)
        self.add("valu", ("vbroadcast", zero_v, zero_const))
        self.add("valu", ("vbroadcast", one_v, one_const))
        self.add("valu", ("vbroadcast", two_v, two_const))
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]))

        hash_c1_v = []
        hash_c3_v = []
        for hi, (_op1, val1, _op2, _op3, val3) in enumerate(HASH_STAGES):
            c1_scalar = self.scratch_const(val1)
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c1_v, c1_scalar))
            c3_scalar = self.scratch_const(val3)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c3_v, c3_scalar))
            hash_c1_v.append(c1_v)
            hash_c3_v.append(c3_v)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))
        body_instrs = []

        def emit_bundle(alu=None, valu=None, load=None, store=None, flow=None, debug=None):
            instr = {}
            if alu:
                instr["alu"] = alu
            if valu:
                instr["valu"] = valu
            if load:
                instr["load"] = load
            if store:
                instr["store"] = store
            if flow:
                instr["flow"] = flow
            if debug:
                instr["debug"] = debug
            if not instr:
                return
            for name, slots in instr.items():
                assert len(slots) <= SLOT_LIMITS[name]
            body_instrs.append(instr)

        def vkeys(round_i, base_i, name):
            return [(round_i, base_i + lane, name) for lane in range(VLEN)]

        def emit_vcompare(vec_addr, keys):
            if not self.enable_vdebug:
                return
            emit_bundle(debug=[("vcompare", vec_addr, keys)])

        # Vector scratch registers (double-buffered)
        buffers = []
        for bi in range(2):
            buffers.append(
                {
                    "idx": self.alloc_scratch(f"idx_v{bi}", VLEN),
                    "val": self.alloc_scratch(f"val_v{bi}", VLEN),
                    "node": self.alloc_scratch(f"node_val_v{bi}", VLEN),
                    "addr": self.alloc_scratch(f"addr_v{bi}", VLEN),
                    "tmp1": self.alloc_scratch(f"tmp1_v{bi}", VLEN),
                    "tmp2": self.alloc_scratch(f"tmp2_v{bi}", VLEN),
                    "cond": self.alloc_scratch(f"cond_v{bi}", VLEN),
                    "idx_addr": self.alloc_scratch(f"idx_addr{bi}"),
                    "val_addr": self.alloc_scratch(f"val_addr{bi}"),
                }
            )

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN
        block_offsets = [self.scratch_const(i) for i in range(0, vector_batch, VLEN)]

        def prefetch_ops(next_buf, next_offset, cycle_idx):
            ops = {"alu": [], "valu": [], "load": []}
            if next_buf is None:
                return ops
            if cycle_idx == 0:
                ops["alu"].append(
                    ("+", next_buf["idx_addr"], self.scratch["inp_indices_p"], next_offset)
                )
                ops["alu"].append(
                    ("+", next_buf["val_addr"], self.scratch["inp_values_p"], next_offset)
                )
            elif cycle_idx == 1:
                ops["load"].append(("vload", next_buf["idx"], next_buf["idx_addr"]))
                ops["load"].append(("vload", next_buf["val"], next_buf["val_addr"]))
            elif cycle_idx == 2:
                ops["valu"].append(("+", next_buf["addr"], next_buf["idx"], forest_base_v))
            elif 3 <= cycle_idx <= 6:
                lane = (cycle_idx - 3) * 2
                ops["load"].append(("load_offset", next_buf["node"], next_buf["addr"], lane))
                ops["load"].append(
                    ("load_offset", next_buf["node"], next_buf["addr"], lane + 1)
                )
            return ops

        for round_i in range(rounds):
            if vector_blocks:
                # Prologue: prefetch block 0 into buffer 0
                buf0 = buffers[0]
                emit_bundle(
                    alu=[
                        (
                            "+",
                            buf0["idx_addr"],
                            self.scratch["inp_indices_p"],
                            block_offsets[0],
                        ),
                        (
                            "+",
                            buf0["val_addr"],
                            self.scratch["inp_values_p"],
                            block_offsets[0],
                        ),
                    ]
                )
                emit_bundle(
                    load=[
                        ("vload", buf0["idx"], buf0["idx_addr"]),
                        ("vload", buf0["val"], buf0["val_addr"]),
                    ]
                )
                emit_bundle(valu=[("+", buf0["addr"], buf0["idx"], forest_base_v)])
                for lane in range(0, VLEN, 2):
                    emit_bundle(
                        load=[
                            ("load_offset", buf0["node"], buf0["addr"], lane),
                            ("load_offset", buf0["node"], buf0["addr"], lane + 1),
                        ]
                    )

                for block in range(vector_blocks):
                    cur = buffers[block % 2]
                    next_block = block + 1
                    next_buf = buffers[next_block % 2] if next_block < vector_blocks else None
                    next_offset = (
                        block_offsets[next_block] if next_block < vector_blocks else None
                    )
                    base_i = block * VLEN

                    # XOR with node values before hashing, while scheduling prefetch
                    pre = prefetch_ops(next_buf, next_offset, 0)
                    emit_bundle(
                        alu=pre["alu"],
                        load=pre["load"],
                        valu=[("^", cur["val"], cur["val"], cur["node"])] + pre["valu"],
                    )

                    cycle_idx = 1
                    for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                        pre = prefetch_ops(next_buf, next_offset, cycle_idx)
                        emit_bundle(
                            alu=pre["alu"],
                            load=pre["load"],
                            valu=[
                                (op1, cur["tmp1"], cur["val"], hash_c1_v[hi]),
                                (op3, cur["tmp2"], cur["val"], hash_c3_v[hi]),
                            ]
                            + pre["valu"],
                        )
                        cycle_idx += 1

                        pre = prefetch_ops(next_buf, next_offset, cycle_idx)
                        emit_bundle(
                            alu=pre["alu"],
                            load=pre["load"],
                            valu=[(op2, cur["val"], cur["tmp1"], cur["tmp2"])]
                            + pre["valu"],
                        )
                        cycle_idx += 1

                    emit_vcompare(cur["val"], vkeys(round_i, base_i, "hashed_val"))

                    # idx_v = 2*idx_v + (1 + (val_v & 1))
                    emit_bundle(
                        valu=[
                            ("&", cur["tmp1"], cur["val"], one_v),
                            ("*", cur["idx"], cur["idx"], two_v),
                        ],
                        store=[("vstore", cur["val_addr"], cur["val"])],
                    )
                    emit_bundle(valu=[("+", cur["tmp1"], cur["tmp1"], one_v)])
                    emit_bundle(valu=[("+", cur["idx"], cur["idx"], cur["tmp1"])])
                    emit_bundle(valu=[("<", cur["cond"], cur["idx"], n_nodes_v)])
                    emit_bundle(flow=[("vselect", cur["idx"], cur["cond"], cur["idx"], zero_v)])
                    emit_bundle(store=[("vstore", cur["idx_addr"], cur["idx"])])

                    emit_vcompare(cur["idx"], vkeys(round_i, base_i, "wrapped_idx"))

            # Scalar tail for any remaining elements
            for i in range(vector_batch, batch_size):
                tail_slots = []
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                tail_slots.append(("load", ("load", tmp_idx, tmp_addr)))
                # val = mem[inp_values_p + i]
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                tail_slots.append(("load", ("load", tmp_val, tmp_addr)))
                # node_val = mem[forest_values_p + idx]
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                )
                tail_slots.append(("load", ("load", tmp_node_val, tmp_addr)))
                # val = myhash(val ^ node_val)
                tail_slots.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                tail_slots.extend(self.build_hash(tmp_val, tmp1, tmp2, round_i, i))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                tail_slots.append(("alu", ("%", tmp1, tmp_val, two_const)))
                tail_slots.append(("alu", ("==", tmp1, tmp1, zero_const)))
                tail_slots.append(
                    ("flow", ("select", tmp3, tmp1, one_const, two_const))
                )
                tail_slots.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                tail_slots.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                # idx = 0 if idx >= n_nodes else idx
                tail_slots.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                tail_slots.append(
                    ("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))
                )
                # mem[inp_indices_p + i] = idx
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                tail_slots.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                tail_slots.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                tail_slots.append(("store", ("store", tmp_addr, tmp_val)))
                body_instrs.extend(self.build(tail_slots))

        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
