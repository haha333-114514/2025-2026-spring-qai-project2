import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator


warnings.filterwarnings("ignore", category=DeprecationWarning)


def data_preprocessing(file_path):
    """Read closing prices and return expected returns and covariance matrix."""
    df = pd.read_excel(file_path)
    data = df.values.astype(float)

    returns = np.zeros((data.shape[0] - 1, data.shape[1]))
    for i in range(data.shape[0] - 1):
        returns[i, :] = (data[i + 1, :] - data[i, :]) / data[i, :]

    exp_ret = np.mean(returns, axis=0)
    cov_mat = np.cov(returns, rowvar=False)
    return exp_ret, cov_mat


def build_bit_metadata(num_assets, bits_per_asset):
    """Return qubit -> asset index and binary weight arrays."""
    asset_index = []
    bit_weight = []
    for asset in range(num_assets):
        for bit in range(bits_per_asset):
            asset_index.append(asset)
            bit_weight.append(2**bit)
    return np.array(asset_index, dtype=int), np.array(bit_weight, dtype=float)


def calc_qubo(num_assets, bits_per_asset, cov_mat, exp_ret, half_q, eta, budget):
    """
    Build QUBO coefficients for multi-bit stock holdings.

    Each stock i is encoded by g bits:
        x_i = b_{i,0} + 2 b_{i,1} + ... + 2^(g-1) b_{i,g-1}

    Objective:
        half_q * x.T @ Sigma @ x - mu.T @ x + eta * (sum(x) - budget)^2
    """
    asset_index, bit_weight = build_bit_metadata(num_assets, bits_per_asset)
    num_qubits = num_assets * bits_per_asset

    linear = np.zeros(num_qubits)
    quadratic = np.zeros((num_qubits, num_qubits))

    for p in range(num_qubits):
        asset_p = asset_index[p]
        weight_p = bit_weight[p]
        linear[p] = (
            half_q * cov_mat[asset_p, asset_p] * weight_p**2
            - exp_ret[asset_p] * weight_p
            + eta * (weight_p**2 - 2.0 * budget * weight_p)
        )

    for p in range(num_qubits):
        asset_p = asset_index[p]
        weight_p = bit_weight[p]
        for q in range(p + 1, num_qubits):
            asset_q = asset_index[q]
            weight_q = bit_weight[q]
            quadratic[p, q] = (
                2.0 * half_q * cov_mat[asset_p, asset_q] * weight_p * weight_q
                + 2.0 * eta * weight_p * weight_q
            )

    constant = eta * budget**2
    return linear, quadratic, constant, asset_index, bit_weight


def qubo_to_ising(linear, quadratic, constant=0.0):
    """Convert QUBO E(b)=a*b+Q*b*b to Ising E(z)=h*z+J*z*z+const."""
    num_qubits = len(linear)
    h = -0.5 * linear.copy()
    J = np.zeros((num_qubits, num_qubits))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            J[i, j] = 0.25 * quadratic[i, j]
            h[i] -= 0.25 * quadratic[i, j]
            h[j] -= 0.25 * quadratic[i, j]

    const = constant + 0.5 * np.sum(linear) + 0.25 * np.sum(quadratic)
    return h, J, const


def calc_h_j(num_assets, bits_per_asset, cov_mat, exp_ret, half_q, eta, budget):
    linear, quadratic, constant, asset_index, bit_weight = calc_qubo(
        num_assets, bits_per_asset, cov_mat, exp_ret, half_q, eta, budget
    )
    h, J, const = qubo_to_ising(linear, quadratic, constant)
    return h, J, const, asset_index, bit_weight


def insert_rx(num_qubits, beta):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(2 * beta, i)
    return qc


def insert_rz(num_qubits, gamma, h):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rz(2 * gamma * h[i], i)
    return qc


def insert_rzz(num_qubits, gamma, J):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if abs(J[i, j]) > 1e-15:
                qc.rzz(2 * gamma * J[i, j], i, j)
    qc.barrier()
    return qc


def insert_h(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc


def get_pauli(num_qubits, index, pauli_type):
    if pauli_type == "Z":
        assert len(index) == 1
        pauli = ["I"] * (num_qubits - 1)
        pauli.insert(index[0], "Z")
        return "".join(pauli)

    if pauli_type == "ZZ":
        assert len(index) == 2
        pauli = ["I"] * (num_qubits - 2)
        for idx in index:
            pauli.insert(idx, "Z")
        return "".join(pauli)

    raise AssertionError("Unsupported Pauli type")


def problem_pauli_operator(num_qubits, h, J):
    pauli_h = PauliSumOp.from_list(
        [(get_pauli(num_qubits, [i], "Z"), h[i]) for i in range(num_qubits)],
        coeff=1.0,
    )

    pauli_j_terms = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if abs(J[i, j]) > 1e-15:
                pauli_j_terms.append((get_pauli(num_qubits, [i, j], "ZZ"), J[i, j]))

    if pauli_j_terms:
        pauli_j = PauliSumOp.from_list(pauli_j_terms, coeff=1.0)
        return pauli_h + pauli_j
    return pauli_h


def one_circuit(num_qubits, h, J, beta, gamma):
    qc = QuantumCircuit(num_qubits)
    qc.append(insert_rz(num_qubits, gamma, h), range(num_qubits))
    qc.append(insert_rzz(num_qubits, gamma, J), range(num_qubits))
    qc.append(insert_rx(num_qubits, beta), range(num_qubits))
    return qc


def get_expectation(compiled_circuit, para_list, hamiltonian, simulator):
    def execute_circ(theta):
        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i + p]] = gamma[i]

        qc_bound = compiled_circuit.assign_parameters(para_dict, inplace=False)
        result = simulator.run(qc_bound).result()
        statevector = result.get_statevector(qc_bound)
        loss = statevector.expectation_value(hamiltonian)
        return np.real(loss)

    return execute_circ


def bitstring_to_statevector(bitstring):
    bitstring = bitstring[::-1]
    dec = int(bitstring, 2)
    state = np.zeros(2 ** len(bitstring))
    state[dec] = 1.0
    return state[None, :]


def decode_holdings(bitstring, num_assets, bits_per_asset):
    holdings = []
    for asset in range(num_assets):
        start = asset * bits_per_asset
        bits = bitstring[start : start + bits_per_asset]
        value = sum((2**k) * int(bits[k]) for k in range(bits_per_asset))
        holdings.append(value)
    return np.array(holdings, dtype=int)


def objective_value(holdings, exp_ret, cov_mat, half_q, eta, budget):
    holdings = holdings.astype(float)
    risk = half_q * holdings @ cov_mat @ holdings
    ret = exp_ret @ holdings
    penalty = eta * (np.sum(holdings) - budget) ** 2
    return risk - ret + penalty


class Callback:
    def __init__(self, step_size):
        self.step_size = step_size
        self.full_values = []
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.full_values.append(value)
        if len(self.full_values) % self.step_size == 0:
            self.values.append(value)
            print(
                f"Iteration {len(self.full_values):4d} | Current Loss: {value:.10f}",
                flush=True,
            )


def print_config(args, num_qubits):
    print("%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%")
    print(
        "budget: %d, assets: %d, g: %d, qubits: %d, eta: %f, layers: %d"
        % (args.budget, args.num_assets, args.g, num_qubits, args.eta, args.layers)
    )
    print("每只股票编码范围: 0 到 %d 股/份" % (2**args.g - 1))


def print_result(
    circuit,
    hamiltonian_matrix,
    para_list,
    solution,
    num_qubits,
    simulator,
    args,
    exp_ret,
    cov_mat,
    h,
    J,
):
    qc = QuantumCircuit(num_qubits)
    p = len(solution) // 2
    beta = solution[:p]
    gamma = solution[p:]

    para_dict = {}
    for i in range(p):
        para_dict[para_list[i]] = beta[i]
        para_dict[para_list[i + p]] = gamma[i]

    qc.append(circuit, range(num_qubits))
    qc.assign_parameters(para_dict, inplace=True)
    circ = transpile(qc, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ).to_dict()

    for key in statevector:
        statevector[key] = np.abs(np.array(statevector[key])) ** 2

    result_sorted = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    basis_states = []
    for bitstring, _ in result_sorted:
        basis_states.append(bitstring_to_statevector(bitstring))
    basis_states = np.concatenate(basis_states, axis=0)
    ising_values = np.sum((basis_states @ hamiltonian_matrix) * basis_states, axis=1)

    decoded_rows = []
    for rank, (raw_bitstring, probability) in enumerate(result_sorted):
        bitstring = raw_bitstring[::-1]
        holdings = decode_holdings(bitstring, args.num_assets, args.g)
        decoded_rows.append(
            {
                "rank": rank,
                "bitstring": bitstring,
                "holdings": holdings,
                "total": int(np.sum(holdings)),
                "ising_value": float(np.real(ising_values[rank])),
                "objective": float(
                    objective_value(
                        holdings,
                        exp_ret,
                        cov_mat,
                        args.half_q,
                        args.eta,
                        args.budget,
                    )
                ),
                "probability": float(probability),
            }
        )

    best_feasible = [row for row in decoded_rows if row["total"] == args.budget]
    best_rows = best_feasible if best_feasible else decoded_rows
    best_row = min(best_rows, key=lambda row: row["objective"])

    print(
        "\nOptimal%s: bits %s, holdings %s, total %d, objective %.8f"
        % (
            " feasible" if best_feasible else "",
            best_row["bitstring"],
            best_row["holdings"].tolist(),
            best_row["total"],
            best_row["objective"],
        )
    )

    print("\n====== 网页图形化界面手填参数指南 ======")
    print(f"当前使用的经典参数: gamma = {solution[args.layers]}, beta = {solution[0]}")
    print("\n1. RZ 门参数：")
    for i in range(num_qubits):
        print(f"  q[{i}] 的 RZ 门 -> {2 * solution[args.layers] * h[i]:.6f}")

    print("\n2. RZZ 门参数：")
    count = 0
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if abs(J[i, j]) > 1e-15:
                count += 1
                print(
                    f"  [{count:02d}] q[{i}] 和 q[{j}] 的 RZZ 门 -> "
                    f"{2 * solution[args.layers] * J[i, j]:.6f}"
                )

    print("\n3. RX 门参数：")
    print(f"  所有 RX 门统一填写 -> {2 * solution[0]:.6f}")

    print("\n----------------- Full result ---------------------", flush=True)
    print("rank\tbits\t\tholdings\ttotal\tobjective\tprobability")
    print("---------------------------------------------------", flush=True)

    value_save = []
    probability_save = []
    selection_save = []
    holdings_save = []

    for row in decoded_rows:
        print(
            "%d\t%-12s\t%-12s\t%d\t%.8f\t%.8f"
            % (
                row["rank"],
                row["bitstring"],
                str(row["holdings"].tolist()),
                row["total"],
                row["objective"],
                row["probability"],
            ),
            flush=True,
        )
        value_save.append(row["objective"])
        probability_save.append(row["probability"])
        selection_save.append(row["bitstring"])
        holdings_save.append(row["holdings"])

    os.makedirs("./output", exist_ok=True)
    output_path = "./output/multibit_budget_{}_g_{}_layers_{}_eta_{}.npz".format(
        args.budget, args.g, args.layers, args.eta
    )
    np.savez(
        output_path,
        value=np.array(value_save),
        probability=np.array(probability_save),
        selection=np.array(selection_save),
        holdings=np.array(holdings_save),
    )
    print(f"\n结果数据已成功保存至 {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=4, help="Total holding budget.")
    parser.add_argument("--num_assets", type=int, default=6, help="The number of assets.")
    parser.add_argument("--g", type=int, default=2, help="Binary bits per asset.")
    parser.add_argument("--half_q", type=float, default=0.25, help="Risk coefficient.")
    parser.add_argument("--eta", type=float, default=6.0, help="Budget penalty coefficient.")
    parser.add_argument("--seed", type=int, default=123456, help="Random seed.")
    parser.add_argument(
        "--optimizer",
        action="store_true",
        default=False,
        help="Use SciPy COBYLA instead of Qiskit SPSA.",
    )
    parser.add_argument("--maxiter", type=int, default=500, help="Max iterations.")
    parser.add_argument("--layers", type=int, default=6, help="The number of QAOA layers.")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/stock_data.xlsx",
        help="Excel file containing stock closing prices.",
    )
    args = parser.parse_args()

    if args.g < 1:
        raise ValueError("g must be at least 1.")

    num_qubits = args.num_assets * args.g
    print_config(args, num_qubits)
    np.random.seed(args.seed)

    print(f"\n正在从 {args.data} 读取股票收盘价并动态计算均值和协方差...")
    try:
        exp_ret, cov_mat = data_preprocessing(args.data)
        if len(exp_ret) < args.num_assets:
            raise ValueError(
                f"数据中只有 {len(exp_ret)} 只股票，少于 --num_assets={args.num_assets}。"
            )
        exp_ret = exp_ret[: args.num_assets]
        cov_mat = cov_mat[: args.num_assets, : args.num_assets]

        scale = np.mean(np.abs(exp_ret))
        if scale <= 0:
            raise ValueError("收益率均值全为 0，无法归一化。")
        exp_ret = exp_ret / scale
        cov_mat = cov_mat / scale
    except Exception as exc:
        print(f"读取或计算失败，请确保表格路径正确且格式规范。错误原因: {exc}")
        raise SystemExit(1)

    print(f"\nNormalization scale = {scale:e}")
    print("\n%%%%%%%%%%%%%%%%%%%% Calculated Input Data %%%%%%%%%%%%%%%%%%%%")
    print("Expected Returns (mu):\n", exp_ret)
    print("-" * 50)
    print("Covariance Matrix (Sigma):")
    with np.printoptions(precision=6, suppress=True):
        print(cov_mat)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    h, J, const, asset_index, bit_weight = calc_h_j(
        args.num_assets,
        args.g,
        cov_mat,
        exp_ret,
        args.half_q,
        args.eta,
        args.budget,
    )

    print("%%%%%%%%%%%%%%%%%%%% Multi-bit Encoding %%%%%%%%%%%%%%%%%%%%")
    for q in range(num_qubits):
        print(
            "q[%d] -> stock %d, bit weight %.0f"
            % (q, asset_index[q], bit_weight[q])
        )
    print(f"Ising constant offset: {const:.8f}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    pauli_sum = problem_pauli_operator(num_qubits, h, J)

    simulator = AerSimulator(method="statevector", device="CPU")
    simulator.set_options(
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
        statevector_parallel_threshold=14,
    )

    beta = [Parameter(f"β{i}") for i in range(args.layers)]
    gamma = [Parameter(f"γ{i}") for i in range(args.layers)]
    para_list = beta + gamma

    qc = QuantumCircuit(num_qubits)
    qc.append(insert_h(num_qubits), range(num_qubits))
    for i in range(args.layers):
        qc.append(one_circuit(num_qubits, h, J, beta[i], gamma[i]), range(num_qubits))
    qc.save_statevector()

    qc_compiled = transpile(qc, simulator)
    print("Circuit Initialization Complete! Start Training...\n")

    expectation = get_expectation(qc_compiled, para_list, pauli_sum, simulator)
    init_point = np.random.uniform(0, 0.001 * np.pi, size=args.layers * 2)

    start = time.time()
    if args.optimizer:
        from scipy.optimize import minimize

        print("Using Scipy COBYLA optimizer...")
        res = minimize(
            expectation,
            init_point,
            method="COBYLA",
            options={"maxiter": args.maxiter},
        )
        solution = res.x
    else:
        print("Using Qiskit SPSA optimizer...")
        callback_func = Callback(step_size=1)
        optimizer = SPSA(
            maxiter=args.maxiter,
            blocking=True,
            second_order=True,
            callback=callback_func,
        )
        res = optimizer.optimize(
            num_vars=args.layers * 2,
            objective_function=expectation,
            initial_point=init_point,
        )
        solution = res[0]

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time() - start))

    print_result(
        qc,
        pauli_sum.to_matrix(),
        para_list,
        solution,
        num_qubits,
        simulator,
        args,
        exp_ret,
        cov_mat,
        h,
        J,
    )

    print("\n正在生成量子线路架构图...")
    try:
        qc_draw = QuantumCircuit(num_qubits)
        qc_draw.append(insert_h(num_qubits), range(num_qubits))
        for i in range(args.layers):
            qc_draw.append(
                one_circuit(num_qubits, h, J, solution[i], solution[i + args.layers]),
                range(num_qubits),
            )

        os.makedirs("./output", exist_ok=True)
        fig = qc_draw.decompose().draw(output="mpl", style="iqp")
        output_image = f"./output/qaoa_multibit_circuit_g{args.g}.png"
        fig.savefig(output_image, bbox_inches="tight")
        print(f"线路展开图已成功导出至: {output_image}")
    except Exception as exc:
        print(f"线路图导出失败: {exc}")


if __name__ == "__main__":
    main()
