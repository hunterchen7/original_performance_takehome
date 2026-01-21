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
        self.enable_stats = False
        self.pipe_buffers_override = 19

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
        node0 = self.alloc_scratch("node0")
        node1 = self.alloc_scratch("node1")
        node2 = self.alloc_scratch("node2")
        node0_v = self.alloc_scratch("node0_v", VLEN)
        node1_v = self.alloc_scratch("node1_v", VLEN)
        node2_v = self.alloc_scratch("node2_v", VLEN)
        forest_base_p1 = self.alloc_scratch("forest_base_p1")
        forest_base_p2 = self.alloc_scratch("forest_base_p2")
        self.add("valu", ("vbroadcast", zero_v, zero_const))
        self.add("valu", ("vbroadcast", one_v, one_const))
        self.add("valu", ("vbroadcast", two_v, two_const))
        self.add("valu", ("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", forest_base_v, self.scratch["forest_values_p"]))
        self.add("load", ("load", node0, self.scratch["forest_values_p"]))
        self.add("alu", ("+", forest_base_p1, self.scratch["forest_values_p"], one_const))
        self.add("alu", ("+", forest_base_p2, self.scratch["forest_values_p"], two_const))
        self.add("load", ("load", node1, forest_base_p1))
        self.add("load", ("load", node2, forest_base_p2))
        self.add("valu", ("vbroadcast", node0_v, node0))
        self.add("valu", ("vbroadcast", node1_v, node1))
        self.add("valu", ("vbroadcast", node2_v, node2))

        hash_c1_v = []
        hash_c3_v = []
        hash_mul_v = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1_scalar = self.scratch_const(val1)
            c1_v = self.alloc_scratch(f"hash_c1_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c1_v, c1_scalar))
            c3_scalar = self.scratch_const(val3)
            c3_v = self.alloc_scratch(f"hash_c3_v_{hi}", VLEN)
            self.add("valu", ("vbroadcast", c3_v, c3_scalar))
            hash_c1_v.append(c1_v)
            hash_c3_v.append(c3_v)
            mul_v = None
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = (1 + (1 << val3)) % (2**32)
                mul_scalar = self.scratch_const(mul)
                mul_v = self.alloc_scratch(f"hash_mul_v_{hi}", VLEN)
                self.add("valu", ("vbroadcast", mul_v, mul_scalar))
            hash_mul_v.append(mul_v)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))
        body_instrs = []

        # Vector scratch registers (pipelined buffers)
        buffers = []
        vector_batch = (batch_size // VLEN) * VLEN
        vector_blocks = vector_batch // VLEN
        block_offsets = [self.scratch_const(i) for i in range(0, vector_batch, VLEN)]
        tail_consts = [self.scratch_const(i) for i in range(vector_batch, batch_size)]

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        per_buffer = VLEN * 4 + 2
        scratch_left = SCRATCH_SIZE - self.scratch_ptr
        max_buffers = scratch_left // per_buffer if scratch_left > 0 else 0
        pipe_buffers = min(vector_blocks, max_buffers) if vector_blocks else 0
        if self.pipe_buffers_override is not None:
            pipe_buffers = min(pipe_buffers, self.pipe_buffers_override)
        # Reuse addr/node scratch for hash/update temps to reduce per-buffer footprint.
        for bi in range(pipe_buffers):
            idx = self.alloc_scratch(f"idx_v{bi}", VLEN)
            val = self.alloc_scratch(f"val_v{bi}", VLEN)
            node = self.alloc_scratch(f"node_val_v{bi}", VLEN)
            addr = self.alloc_scratch(f"addr_v{bi}", VLEN)
            idx_addr = self.alloc_scratch(f"idx_addr{bi}")
            val_addr = self.alloc_scratch(f"val_addr{bi}")
            buffers.append(
                {
                    "idx": idx,
                    "val": val,
                    "node": node,
                    "addr": addr,
                    "tmp1": addr,
                    "tmp2": node,
                    "cond": node,
                    "idx_addr": idx_addr,
                    "val_addr": val_addr,
                }
            )

        def schedule_all_rounds():
            if vector_blocks == 0:
                return []
            instrs = []
            active = []
            free_bufs = list(range(pipe_buffers))
            next_block = 0

            def start_block():
                nonlocal next_block
                if next_block >= vector_blocks or not free_bufs:
                    return False
                buf_idx = free_bufs.pop(0)
                active.append(
                    {
                        "block": next_block,
                        "buf_idx": buf_idx,
                        "buf": buffers[buf_idx],
                        "offset": block_offsets[next_block],
                        "phase": "init_addr",
                        "round": 0,
                        "stage": 0,
                        "gather": 0,
                        "use_node0": True,
                        "store_val_pending": False,
                    }
                )
                next_block += 1
                return True

            while free_bufs and next_block < vector_blocks:
                start_block()

            while active or next_block < vector_blocks:
                while free_bufs and next_block < vector_blocks:
                    start_block()

                alu_ops = []
                load_ops = []
                valu_ops = []
                store_ops = []
                flow_ops = []

                alu_slots = SLOT_LIMITS["alu"]
                load_slots = SLOT_LIMITS["load"]
                valu_slots = SLOT_LIMITS["valu"]
                store_slots = SLOT_LIMITS["store"]
                flow_slots = SLOT_LIMITS["flow"]

                scheduled_main = set()

                # Flow: vselect for bounds
                if flow_slots:
                    for block in active:
                        if block["phase"] == "update5":
                            buf = block["buf"]
                            if block["round"] == 0 and rounds > 1:
                                flow_ops.append(
                                    (
                                        "vselect",
                                        buf["node"],
                                        buf["cond"],
                                        node2_v,
                                        node1_v,
                                    )
                                )
                                block["round"] = 1
                                block["stage"] = 0
                                block["gather"] = 0
                                block["next_phase"] = "xor"
                            else:
                                flow_ops.append(
                                    ("vselect", buf["idx"], buf["cond"], buf["idx"], zero_v)
                                )
                                if block["round"] + 1 < rounds:
                                    block["round"] += 1
                                    block["stage"] = 0
                                    block["gather"] = 0
                                    block["next_phase"] = "addr"
                                else:
                                    block["next_phase"] = "store_idx"
                            scheduled_main.add(block["block"])
                            flow_slots -= 1
                            break

                # Store val after the final hash stage (can overlap with updates).
                for block in active:
                    if store_slots == 0:
                        break
                    if block["store_val_pending"]:
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["val_addr"], buf["val"]))
                        block["store_val_pending"] = False
                        store_slots -= 1

                # Store idx after updates, once val is stored (finishes blocks).
                for block in active:
                    if store_slots == 0:
                        break
                    if (
                        block["phase"] == "store_idx"
                        and not block["store_val_pending"]
                        and block["block"] not in scheduled_main
                    ):
                        buf = block["buf"]
                        store_ops.append(("vstore", buf["idx_addr"], buf["idx"]))
                        block["next_phase"] = "done"
                        scheduled_main.add(block["block"])
                        store_slots -= 1


                # Loads: prioritize vload to feed the pipeline
                if load_slots >= 2:
                    for block in active:
                        if block["phase"] == "vload" and block["block"] not in scheduled_main:
                            buf = block["buf"]
                            load_ops.append(("vload", buf["idx"], buf["idx_addr"]))
                            load_ops.append(("vload", buf["val"], buf["val_addr"]))
                            block["next_phase"] = "xor"
                            scheduled_main.add(block["block"])
                            load_slots -= 2
                            break

                # Loads: gather node values
                for block in active:
                    if load_slots == 0:
                        break
                    if block["phase"] == "gather" and block["block"] not in scheduled_main:
                        buf = block["buf"]
                        for _ in range(load_slots):
                            lane = block["gather"]
                            if lane >= VLEN:
                                break
                            load_ops.append(("load_offset", buf["node"], buf["addr"], lane))
                            block["gather"] += 1
                            load_slots -= 1
                            if load_slots == 0:
                                break
                        if block["gather"] >= VLEN:
                            block["next_phase"] = "xor"
                            scheduled_main.add(block["block"])

                # VALU tasks with priorities
                valu_tasks = []
                for block in active:
                    if block["block"] in scheduled_main:
                        continue
                    phase = block["phase"]
                    if phase == "addr":
                        valu_tasks.append((0, 1, block))
                    elif phase == "xor":
                        valu_tasks.append((1, 1, block))
                    elif phase == "hash_op13":
                        stage = block["stage"]
                        cost = 1 if hash_mul_v[stage] is not None else 2
                        valu_tasks.append((2, cost, block))
                    elif phase == "hash_op2":
                        valu_tasks.append((3, 1, block))
                    elif phase == "update1":
                        valu_tasks.append((4, 2, block))
                    elif phase == "update2":
                        valu_tasks.append((5, 1, block))
                    elif phase == "update3":
                        valu_tasks.append((6, 1, block))

                valu_tasks.sort(key=lambda x: x[0])
                for _prio, cost, block in valu_tasks:
                    if valu_slots < cost:
                        continue
                    if block["block"] in scheduled_main:
                        continue
                    buf = block["buf"]
                    phase = block["phase"]
                    if phase == "hash_op2":
                        hi = block["stage"]
                        op2 = HASH_STAGES[hi][2]
                        valu_ops.append((op2, buf["val"], buf["tmp1"], buf["tmp2"]))
                        if hi + 1 == len(HASH_STAGES):
                            if block["round"] + 1 == rounds:
                                block["store_val_pending"] = True
                            block["next_phase"] = "update1"
                        else:
                            block["stage"] = hi + 1
                            block["next_phase"] = "hash_op13"
                        scheduled_main.add(block["block"])
                        valu_slots -= 1
                    elif phase == "hash_op13":
                        hi = block["stage"]
                        mul_v = hash_mul_v[hi]
                        if mul_v is not None:
                            valu_ops.append(
                                ("multiply_add", buf["val"], buf["val"], mul_v, hash_c1_v[hi])
                            )
                            if hi + 1 == len(HASH_STAGES):
                                if block["round"] + 1 == rounds:
                                    block["store_val_pending"] = True
                                block["next_phase"] = "update1"
                            else:
                                block["stage"] = hi + 1
                                block["next_phase"] = "hash_op13"
                            scheduled_main.add(block["block"])
                            valu_slots -= 1
                        else:
                            op1 = HASH_STAGES[hi][0]
                            op3 = HASH_STAGES[hi][3]
                            valu_ops.append((op1, buf["tmp1"], buf["val"], hash_c1_v[hi]))
                            valu_ops.append((op3, buf["tmp2"], buf["val"], hash_c3_v[hi]))
                            block["next_phase"] = "hash_op2"
                            scheduled_main.add(block["block"])
                            valu_slots -= 2
                    elif phase == "xor":
                        if block["use_node0"]:
                            valu_ops.append(("^", buf["val"], buf["val"], node0_v))
                            block["use_node0"] = False
                        else:
                            valu_ops.append(("^", buf["val"], buf["val"], buf["node"]))
                        block["next_phase"] = "hash_op13"
                        scheduled_main.add(block["block"])
                        valu_slots -= 1
                    elif phase == "update1":
                        valu_ops.append(("&", buf["tmp1"], buf["val"], one_v))
                        valu_ops.append(
                            (
                                "multiply_add",
                                buf["idx"],
                                buf["idx"],
                                two_v,
                                one_v,
                            )
                        )
                        block["next_phase"] = "update2"
                        scheduled_main.add(block["block"])
                        valu_slots -= 2
                    elif phase == "update2":
                        valu_ops.append(("+", buf["idx"], buf["idx"], buf["tmp1"]))
                        block["next_phase"] = "update3"
                        scheduled_main.add(block["block"])
                        valu_slots -= 1
                    elif phase == "update3":
                        if block["round"] == 0 and rounds > 1:
                            valu_ops.append(("+", buf["cond"], buf["tmp1"], zero_v))
                        else:
                            valu_ops.append(("<", buf["cond"], buf["idx"], n_nodes_v))
                        block["next_phase"] = "update5"
                        scheduled_main.add(block["block"])
                        valu_slots -= 1
                    elif phase == "addr":
                        valu_ops.append(("+", buf["addr"], buf["idx"], forest_base_v))
                        block["next_phase"] = "gather"
                        scheduled_main.add(block["block"])
                        valu_slots -= 1

                # ALU tasks: compute base addresses once per block
                for block in active:
                    if alu_slots < 2:
                        break
                    if block["phase"] == "init_addr" and block["block"] not in scheduled_main:
                        buf = block["buf"]
                        alu_ops.append(("+", buf["idx_addr"], self.scratch["inp_indices_p"], block["offset"]))
                        alu_ops.append(("+", buf["val_addr"], self.scratch["inp_values_p"], block["offset"]))
                        block["next_phase"] = "vload"
                        scheduled_main.add(block["block"])
                        alu_slots -= 2

                if not (alu_ops or load_ops or valu_ops or store_ops or flow_ops):
                    raise RuntimeError("scheduler made no progress")

                instr = {}
                if alu_ops:
                    instr["alu"] = alu_ops
                if load_ops:
                    instr["load"] = load_ops
                if valu_ops:
                    instr["valu"] = valu_ops
                if store_ops:
                    instr["store"] = store_ops
                if flow_ops:
                    instr["flow"] = flow_ops
                instrs.append(instr)

                # Apply state transitions
                new_active = []
                for block in active:
                    next_phase = block.pop("next_phase", None)
                    if next_phase:
                        block["phase"] = next_phase
                    if block["phase"] == "done":
                        free_bufs.append(block["buf_idx"])
                    else:
                        new_active.append(block)
                active = new_active

            return instrs

        body_instrs.extend(schedule_all_rounds())
        for round_i in range(rounds):
            # Scalar tail for any remaining elements
            for i, i_const in zip(range(vector_batch, batch_size), tail_consts):
                tail_slots = []
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
