import os, sys, json, math, hashlib, subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
SUB_NX = 256
SUB_NY = 256
STEPS = 400
LAMBDA_DAMP = 0.15

def get_gpu_uuid() -> str:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], encoding="utf-8")
    return out.strip().splitlines()[0].strip()

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)
    neighbors = {"up": nbr_up, "down": nbr_down, "left": nbr_left, "right": nbr_right}
    return cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed

def initialize_subdomains(coords, neighbors, instance_seed):
    row_rank, col_rank = coords
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)
    u = np.zeros((SUB_NY + 2, SUB_NX + 2), dtype=np.float64)
    kappa = np.zeros((SUB_NY + 2, SUB_NX + 2), dtype=np.float64)
    source = np.zeros((SUB_NY + 2, SUB_NX + 2), dtype=np.float64)
    rng = np.random.default_rng(instance_seed)
    pulses = [(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.5, 1.5), rng.uniform(0.04, 0.08)) for _ in range(4)]
    src_x, src_y = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    global_i = row_rank * SUB_NY + np.arange(SUB_NY + 2)
    global_j = col_rank * SUB_NX + np.arange(SUB_NX + 2)
    Y, X = np.meshgrid(global_i * dy, global_j * dx, indexing="ij")
    kappa[:] = np.clip(0.16 + 0.08 * np.sin(2.0 * math.pi * X) * np.cos(2.0 * math.pi * Y), 0.08, 0.24)
    for px, py, amp, sigma in pulses:
        u += amp * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2.0 * sigma ** 2))
    source[:] = 0.5 * np.exp(-((X - src_x) ** 2 + (Y - src_y) ** 2) / (2.0 * 0.05 ** 2))
    if neighbors["up"] == MPI.PROC_NULL: u[:2, :] = 0.0
    if neighbors["down"] == MPI.PROC_NULL: u[SUB_NY:, :] = 0.0
    if neighbors["left"] == MPI.PROC_NULL: u[:, :2] = 0.0
    if neighbors["right"] == MPI.PROC_NULL: u[:, SUB_NX:] = 0.0
    return u, kappa, source, dx, dy

@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[ny, idx + 1]
    if idx < ny:
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]

@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right, has_up, has_down, has_left, has_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_up[idx] if has_up else 0.0
        u[ny + 1, idx + 1] = recv_down[idx] if has_down else 0.0
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx] if has_left else 0.0
        u[idx + 1, nx + 1] = recv_right[idx] if has_right else 0.0

def post_mpi_halos(cart_comm, neighbors, buffers):
    reqs = []
    if neighbors["up"] != MPI.PROC_NULL:
        reqs += [cart_comm.Irecv(buffers["h_recv_up"], source=neighbors["up"], tag=1), cart_comm.Isend(buffers["h_send_up"], dest=neighbors["up"], tag=0)]
    if neighbors["down"] != MPI.PROC_NULL:
        reqs += [cart_comm.Irecv(buffers["h_recv_down"], source=neighbors["down"], tag=0), cart_comm.Isend(buffers["h_send_down"], dest=neighbors["down"], tag=1)]
    if neighbors["left"] != MPI.PROC_NULL:
        reqs += [cart_comm.Irecv(buffers["h_recv_left"], source=neighbors["left"], tag=3), cart_comm.Isend(buffers["h_send_left"], dest=neighbors["left"], tag=2)]
    if neighbors["right"] != MPI.PROC_NULL:
        reqs += [cart_comm.Irecv(buffers["h_recv_right"], source=neighbors["right"], tag=2), cart_comm.Isend(buffers["h_send_right"], dest=neighbors["right"], tag=3)]
    return reqs

@cuda.jit
def tiled_stencil_kernel(u, u_next, kappa, source, dt, inv_dx2, inv_dy2, lam, r_start, r_end, c_start, c_end, wall_up, wall_down, wall_left, wall_right):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    sy, sx = ty + 1, tx + 1
    smem[sy, sx] = u[i, j] if (i <= SUB_NY + 1 and j <= SUB_NX + 1) else 0.0
    if ty == 0: smem[0, sx] = u[i - 1, j] if (i - 1 >= 0 and j <= SUB_NX + 1) else 0.0
    if ty == TILE_DIM - 1: smem[SHARED_DIM - 1, sx] = u[i + 1, j] if (i + 1 <= SUB_NY + 1 and j <= SUB_NX + 1) else 0.0
    if tx == 0: smem[sy, 0] = u[i, j - 1] if (j - 1 >= 0 and i <= SUB_NY + 1) else 0.0
    if tx == TILE_DIM - 1: smem[sy, SHARED_DIM - 1] = u[i, j + 1] if (j + 1 <= SUB_NX + 1 and i <= SUB_NY + 1) else 0.0
    cuda.syncthreads()
    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        if (wall_up and i == 1) or (wall_down and i == SUB_NY) or (wall_left and j == 1) or (wall_right and j == SUB_NX):
            u_next[i, j] = 0.0
            return
        u_c, u_n, u_s, u_w, u_e = smem[sy, sx], smem[sy - 1, sx], smem[sy + 1, sx], smem[sy, sx - 1], smem[sy, sx + 1]
        k_c = kappa[i, j]
        k_iph = 0.5 * (kappa[i + 1, j] + k_c)
        k_imh = 0.5 * (k_c + kappa[i - 1, j])
        k_jph = 0.5 * (kappa[i, j + 1] + k_c)
        k_jmh = 0.5 * (k_c + kappa[i, j - 1])
        diff_y = (k_iph * (u_s - u_c) - k_imh * (u_c - u_n)) * inv_dy2
        diff_x = (k_jph * (u_e - u_c) - k_jmh * (u_c - u_w)) * inv_dx2
        react = -lam * (u_c * u_c * u_c) + source[i, j]
        u_next[i, j] = u_c + dt * (diff_y + diff_x + react)

def launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, r0, r1, c0, c1, stream):
    rows, cols = r1 - r0 + 1, c1 - c0 + 1
    if rows <= 0 or cols <= 0: return
    grid = ((cols + TILE_DIM - 1) // TILE_DIM, (rows + TILE_DIM - 1) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, LAMBDA_DAMP, r0, r1, c0, c1, *walls)

def step_serialized(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs, stream):
    pack_halos_kernel[1, 256, stream](d_u, bufs["d_send_up"], bufs["d_send_down"], bufs["d_send_left"], bufs["d_send_right"], SUB_NY, SUB_NX)
    for k in ("up", "down", "left", "right"): bufs[f"d_send_{k}"].copy_to_host(bufs[f"h_send_{k}"], stream=stream)
    stream.synchronize()
    MPI.Request.Waitall(post_mpi_halos(cart_comm, neighbors, bufs))
    for k in ("up", "down", "left", "right"):
        if neighbors[k] != MPI.PROC_NULL: bufs[f"d_recv_{k}"].copy_to_device(bufs[f"h_recv_{k}"], stream=stream)
    unpack_halos_kernel[1, 256, stream](d_u, bufs["d_recv_up"], bufs["d_recv_down"], bufs["d_recv_left"], bufs["d_recv_right"], neighbors["up"] != MPI.PROC_NULL, neighbors["down"] != MPI.PROC_NULL, neighbors["left"] != MPI.PROC_NULL, neighbors["right"] != MPI.PROC_NULL, SUB_NY, SUB_NX)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, 1, SUB_NY, 1, SUB_NX, stream)
    stream.synchronize()

def step_overlapped(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs, stream_boundary, stream_interior):
    pack_halos_kernel[1, 256, stream_boundary](d_u, bufs["d_send_up"], bufs["d_send_down"], bufs["d_send_left"], bufs["d_send_right"], SUB_NY, SUB_NX)
    for k in ("up", "down", "left", "right"): bufs[f"d_send_{k}"].copy_to_host(bufs[f"h_send_{k}"], stream=stream_boundary)
    stream_boundary.synchronize()
    reqs = post_mpi_halos(cart_comm, neighbors, bufs)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, 2, SUB_NY - 1, 2, SUB_NX - 1, stream_interior)
    MPI.Request.Waitall(reqs)
    for k in ("up", "down", "left", "right"):
        if neighbors[k] != MPI.PROC_NULL: bufs[f"d_recv_{k}"].copy_to_device(bufs[f"h_recv_{k}"], stream=stream_boundary)
    stream_interior.synchronize()
    unpack_halos_kernel[1, 256, stream_boundary](d_u, bufs["d_recv_up"], bufs["d_recv_down"], bufs["d_recv_left"], bufs["d_recv_right"], neighbors["up"] != MPI.PROC_NULL, neighbors["down"] != MPI.PROC_NULL, neighbors["left"] != MPI.PROC_NULL, neighbors["right"] != MPI.PROC_NULL, SUB_NY, SUB_NX)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, 1, 1, 1, SUB_NX, stream_boundary)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, SUB_NY, SUB_NY, 1, SUB_NX, stream_boundary)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, 2, SUB_NY - 1, 1, 1, stream_boundary)
    launch_box(d_u, d_u_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, 2, SUB_NY - 1, SUB_NX, SUB_NX, stream_boundary)
    stream_boundary.synchronize()

def compute_global_diagnostics(cart_comm, u_host, u_prev_host, dx, dy, dt):
    local_energy = float(np.sum(u_host[1:-1, 1:-1]) * dx * dy)
    diff = (u_host[1:-1, 1:-1] - u_prev_host[1:-1, 1:-1]) / dt
    local_res_sq = float(np.sum(diff * diff) * dx * dy)
    return cart_comm.allreduce(local_energy, op=MPI.SUM), math.sqrt(cart_comm.allreduce(local_res_sq, op=MPI.SUM))

def run_solver(github_user: str):
    cart_comm, rank, coords, neighbors, gpu_uuid, instance_seed = init_distributed_context(github_user)
    u_init, kappa, source, dx, dy = initialize_subdomains(coords, neighbors, instance_seed)
    global_kmax = cart_comm.allreduce(float(np.max(kappa)), op=MPI.MAX)
    dt_max = (dx * dx) / (4.0 * global_kmax)
    dt = 0.45 * dt_max
    inv_dx2, inv_dy2 = 1.0 / (dx * dx), 1.0 / (dy * dy)
    walls = tuple(neighbors[k] == MPI.PROC_NULL for k in ("up", "down", "left", "right"))

    def make_buffers():
        return {f"{p}_{d}_{k}": (cuda.pinned_array(SUB_NX, dtype=np.float64) if p == "h" else cuda.device_array(SUB_NX, dtype=np.float64)) for p in ("h", "d") for d in ("send", "recv") for k in ("up", "down", "left", "right")}

    d_kappa, d_source = cuda.to_device(kappa), cuda.to_device(source)
    bufs_ser, stream_ser = make_buffers(), cuda.stream()
    d_u_tmp, d_u_tmp2 = cuda.to_device(u_init.copy()), cuda.to_device(u_init.copy())
    step_serialized(d_u_tmp, d_u_tmp2, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs_ser, stream_ser)

    d_u_ser, d_u_ser_next = cuda.to_device(u_init.copy()), cuda.to_device(u_init.copy())
    cart_comm.Barrier()

    ev_start_ser, ev_end_ser = cuda.event(), cuda.event()
    ev_start_ser.record(stream_ser)
    for _ in range(STEPS):
        step_serialized(d_u_ser, d_u_ser_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs_ser, stream_ser)
        d_u_ser, d_u_ser_next = d_u_ser_next, d_u_ser
    ev_end_ser.record(stream_ser); ev_end_ser.synchronize()
    serialized_ms = cuda.event_elapsed_time(ev_start_ser, ev_end_ser)
    u_final_ser = d_u_ser.copy_to_host()

    bufs_ov, stream_boundary, stream_interior = make_buffers(), cuda.stream(), cuda.stream()
    d_u_tmp3, d_u_tmp4 = cuda.to_device(u_init.copy()), cuda.to_device(u_init.copy())
    step_overlapped(d_u_tmp3, d_u_tmp4, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs_ov, stream_boundary, stream_interior)

    d_u_ov, d_u_ov_next = cuda.to_device(u_init.copy()), cuda.to_device(u_init.copy())
    cart_comm.Barrier()

    energy_trace = []
    e0, r0 = compute_global_diagnostics(cart_comm, u_init, u_init, dx, dy, dt)
    energy_trace.append({"step": 0, "energy": e0, "l2_residual": r0})

    ev_start_ov, ev_end_ov = cuda.event(), cuda.event()
    ev_start_ov.record(stream_boundary)
    for step in range(1, STEPS + 1):
        step_overlapped(d_u_ov, d_u_ov_next, d_kappa, d_source, dt, inv_dx2, inv_dy2, walls, cart_comm, neighbors, bufs_ov, stream_boundary, stream_interior)
        d_u_ov, d_u_ov_next = d_u_ov_next, d_u_ov
        if step % 100 == 0:
            en, res = compute_global_diagnostics(cart_comm, d_u_ov.copy_to_host(), d_u_ov_next.copy_to_host(), dx, dy, dt)
            energy_trace.append({"step": step, "energy": en, "l2_residual": res})
    ev_end_ov.record(stream_boundary); ev_end_ov.synchronize()
    overlapped_ms = cuda.event_elapsed_time(ev_start_ov, ev_end_ov)
    u_final_ov = d_u_ov.copy_to_host()

    parity_max_diff = cart_comm.allreduce(float(np.max(np.abs(u_final_ser[1:-1, 1:-1] - u_final_ov[1:-1, 1:-1]))), op=MPI.MAX)
    max_ser_ms = cart_comm.allreduce(serialized_ms, op=MPI.MAX)
    max_ov_ms = cart_comm.allreduce(overlapped_ms, op=MPI.MAX)

    if rank == 0:
        ptx_code = next(iter(tiled_stencil_kernel.inspect_asm().values()))
        st_c, ld_c, bar_c = ptx_code.count("st.shared.f64"), ptx_code.count("ld.shared.f64"), ptx_code.count("bar.sync")
        ptx_ok = (st_c > 0) and (ld_c > 0) and (bar_c > 0)
        manifest = {"github_username": github_user, "gpu_uuid": gpu_uuid, "instance_seed": instance_seed, "mpi_ranks": 4, "grid_topology": [2, 2], "dx": dx, "kappa_max": global_kmax, "dt_max": dt_max, "dt": dt, "serialized_ms": round(max_ser_ms, 3), "overlapped_ms": round(max_ov_ms, 3), "parity_max_diff": parity_max_diff, "ptx_counts": {"st.shared.f64": st_c, "ld.shared.f64": ld_c, "bar.sync": bar_c}, "ptx_audit_passed": ptx_ok, "energy_trace": energy_trace}
        with open("submission_manifest.json", "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
        print("=== DISTRIBUTED MPI + CUDA SOLVER TELEMETRY ===")
        print(f"GITHUB_USERNAME  : {github_user}\nGPU_UUID         : {gpu_uuid}\nINSTANCE_SEED    : {instance_seed}\nGrid dx          : {dx:.8e}\nkappa_max        : {global_kmax:.6f}\ndt_max (CFL)     : {dt_max:.8e}\ndt (used)        : {dt:.8e}\nserialized_ms    : {max_ser_ms:.3f} ms\noverlapped_ms    : {max_ov_ms:.3f} ms\nparity_max_diff  : {parity_max_diff:.3e}\nPTX st.shared.f64: {st_c}\nPTX ld.shared.f64: {ld_c}\nPTX bar.sync     : {bar_c}\nptx_audit_passed : {ptx_ok}\n\n--- Energy & L2 Residual Trace ---")
        for item in energy_trace: print(f"Step {item['step']:3d} | Energy = {item['energy']:.8f} | L2 Residual = {item['l2_residual']:.8e}")

if __name__ == "__main__":
    if len(sys.argv) < 2: raise ValueError("Usage: python3 distributed_cuda_solver.py <GITHUB_USERNAME>")
    run_solver(sys.argv[1])
