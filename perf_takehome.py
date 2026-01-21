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
        vlen_const = self.scratch_const(VLEN)

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

        body = []  # array of slots

        # Vector scratch registers
        idx_v = self.alloc_scratch("idx_v", VLEN)
        val_v = self.alloc_scratch("val_v", VLEN)
        node_val_v = self.alloc_scratch("node_val_v", VLEN)
        addr_v = self.alloc_scratch("addr_v", VLEN)
        tmp1_v = self.alloc_scratch("tmp1_v", VLEN)
        tmp2_v = self.alloc_scratch("tmp2_v", VLEN)
        tmp3_v = self.alloc_scratch("tmp3_v", VLEN)
        cond_v = self.alloc_scratch("cond_v", VLEN)

        # Scalar scratch registers
        idx_addr = self.alloc_scratch("idx_addr")
        val_addr = self.alloc_scratch("val_addr")
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        def build_hash_vec(val_hash_addr, tmp1_v, tmp2_v):
            slots = []
            for hi, (op1, _val1, op2, op3, _val3) in enumerate(HASH_STAGES):
                slots.append(("valu", (op1, tmp1_v, val_hash_addr, hash_c1_v[hi])))
                slots.append(("valu", (op3, tmp2_v, val_hash_addr, hash_c3_v[hi])))
                slots.append(("valu", (op2, val_hash_addr, tmp1_v, tmp2_v)))
            return slots

        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN

        for round in range(rounds):
            # Initialize vector base addresses for this round
            body.append(("alu", ("+", idx_addr, self.scratch["inp_indices_p"], zero_const)))
            body.append(("alu", ("+", val_addr, self.scratch["inp_values_p"], zero_const)))

            for vi in range(vector_blocks):
                # idx_v = mem[inp_indices_p + i : i + VLEN]
                body.append(("load", ("vload", idx_v, idx_addr)))
                # val_v = mem[inp_values_p + i : i + VLEN]
                body.append(("load", ("vload", val_v, val_addr)))

                # node_val_v = mem[forest_values_p + idx_v] (gather)
                body.append(("valu", ("+", addr_v, idx_v, forest_base_v)))
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", node_val_v, addr_v, lane)))

                # val_v = myhash(val_v ^ node_val_v)
                body.append(("valu", ("^", val_v, val_v, node_val_v)))
                body.extend(build_hash_vec(val_v, tmp1_v, tmp2_v))

                # idx_v = 2*idx_v + (1 if val_v % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1_v, val_v, two_v)))
                body.append(("valu", ("==", cond_v, tmp1_v, zero_v)))
                body.append(("flow", ("vselect", tmp3_v, cond_v, one_v, two_v)))
                body.append(("valu", ("*", idx_v, idx_v, two_v)))
                body.append(("valu", ("+", idx_v, idx_v, tmp3_v)))
                # idx_v = 0 if idx_v >= n_nodes else idx_v
                body.append(("valu", ("<", cond_v, idx_v, n_nodes_v)))
                body.append(("flow", ("vselect", idx_v, cond_v, idx_v, zero_v)))

                # Store updated vectors
                body.append(("store", ("vstore", idx_addr, idx_v)))
                body.append(("store", ("vstore", val_addr, val_v)))

                # Advance base addresses unless this is the last vector block
                if vi + 1 < vector_blocks:
                    body.append(("alu", ("+", idx_addr, idx_addr, vlen_const)))
                    body.append(("alu", ("+", val_addr, val_addr, vlen_const)))

            # Scalar tail for any remaining elements
            for i in range(vector_batch, batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
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
