import argparse
import time
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import SPSA, COBYLA

# 静音新版本 Qiskit 的迁移警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

def print_config():
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, eta: %f, layers: %d' % (budget, num_slices, eta, layers))

def data_preprocessing(file_path):
    '''
    动态预处理：读取 Excel 收盘价并计算期望收益向量和协方差矩阵
    '''
    # excel --> DataFrame
    df = pd.read_excel(file_path)
    data = df.values
    
    # 收盘价转化为每日收益率 (Rate of Return)
    RoR = np.zeros([data.shape[0]-1, data.shape[1]])   
    for i in range(data.shape[0]-1):
        RoR[i,:] = (data[i+1,:] - data[i,:]) / data[i,:]
        
    # 计算每只股票收益率的均值，得到期望收益向量 mu
    exp_ret = np.mean(RoR, axis=0)
    # 计算资产间的协方差矩阵 Sigma (rowvar=False 代表列为资产)
    cov_mat = np.cov(RoR, rowvar=False)

    return exp_ret, cov_mat

def calc_J(num_qubits, half_q, cov_mat, eta):
    '''计算 Rzz 门系数矩阵 '''
    J = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            J[i][j] = 0.25 * (2 * half_q) * cov_mat[i][j] + 0.5 * eta
    return J

def calc_h(num_qubits, num_assets, half_q, cov_mat, exp_ret, eta, budget):
    '''计算 Rz 门系数向量 '''
    h = np.zeros(num_qubits)
    for i in range(num_qubits):
        sum_sigma_ij = np.sum(cov_mat[i])
        h[i] = -0.25 * (2 * half_q) * sum_sigma_ij + 0.5 * exp_ret[i] + eta * (-num_assets / 2.0 + budget)
    return h

def insert_RX(num_qubits, beta): 
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(2 * beta, i)
    return qc

def insert_RZ(num_qubits, gamma, h): 
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rz(2 * gamma * h[i], i)
    return qc

def insert_RZZ(num_qubits, gamma, J):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qc.rzz(2 * gamma * J[i][j], i, j)
    qc.barrier()
    return qc

def insert_H(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc

def get_Pauli(num_qubits, index, type):
    if type == 'Z':
        assert len(index) == 1
        index = index[0]
        _Pauli = ['I'] * (num_qubits - 1)
        _Pauli.insert(index, 'Z')
        return ''.join(_Pauli)
    elif type == 'ZZ':
        assert len(index) == 2
        _Pauli = ['I'] * (num_qubits - 2)
        for i in range(len(index)):
            _Pauli.insert(index[i], 'Z')
        return ''.join(_Pauli)
    else:
        raise AssertionError()

def problem_PauliOperator(num_qubits, h, J):
    Pauli_h_list = []
    for i in range(num_qubits):
        Pauli_h_list.append((get_Pauli(num_qubits, [i], 'Z'), h[i]))
    Pauli_h = PauliSumOp.from_list(Pauli_h_list, coeff=1.0)

    Pauli_J_list = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli_J_list.append((get_Pauli(num_qubits, [i,j], 'ZZ'), J[i][j]))
    Pauli_J = PauliSumOp.from_list(Pauli_J_list, coeff=1.0)

    return Pauli_h + Pauli_J

def oneCircuit(num_qubits, h, J, beta, gamma):
    qc = QuantumCircuit(num_qubits)
    qc.append(insert_RZ(num_qubits, gamma, h), range(num_qubits))
    qc.append(insert_RZZ(num_qubits, gamma, J), range(num_qubits))
    qc.append(insert_RX(num_qubits, beta), range(num_qubits))
    return qc

def get_expectation(compiled_circuit, para_list, Hamiltonian, num_qubits, simulator):
    ''' 优化性能：接收编译好的线路，仅做动态参数绑定 '''
    def execute_circ(theta):
        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i+p]] = gamma[i]
        
        qc_bound = compiled_circuit.assign_parameters(para_dict, inplace=False)
        result = simulator.run(qc_bound).result()
        _statevector = result.get_statevector(qc_bound)

        loss = _statevector.expectation_value(Hamiltonian)
        return np.real(loss)

    return execute_circ

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

class callback:
    ''' 实时在控制台打印每一步迭代的 Loss '''
    def __init__(self, step_size: int):
        self.step_size = step_size
        self.full_values = []
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.full_values.append(value)
        if len(self.full_values) % self.step_size == 0:
            self.values.append(value)
            print(f"Iteration {len(self.full_values):4d} | Current Loss: {value:.10f}", flush=True)

def print_result(circuit, Hamiltonian, para_list, solution, num_qubits, simulator, budget, layers, eta):
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
    
    for i in statevector:
        statevector[i] = np.abs(np.array(statevector[i])) ** 2
        
    result_sorted = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    
    mm = []
    for i in range(len(result_sorted)):
        x, _ = result_sorted[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)
    value_mm = np.sum((mm @ Hamiltonian) * mm, axis=1)

    min_index = np.argmin(value_mm)
    print("\nOptimal: selection {}, value {:.8f}".format(result_sorted[min_index][0][::-1], value_mm[min_index]))

    # 可以在代码里临时加一段这个逻辑来帮你“抄作业”
    print("\n====== 网页图形化界面手填参数指南 ======")
    print(f"当前使用的经典参数: gamma = {solution[layers]}, beta = {solution[0]}")
    print("\n1. 第 2 列所有 RZ 门应该填入的值：")
    for i in range(num_qubits):
        print(f"  q[{i}] 的 RZ 门 -> {2 * solution[layers] * h[i]:.6f}")

    print("\n2. 中间所有夹心 RZ 门（J 矩阵全连接）应该填入的值：")
    count = 0
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            count += 1
            val = 2 * solution[layers] * J[i][j]
            print(f"  [{count:02d}] q[{i}] 和 q[{j}] 之间的夹心 RZ 门 -> {val:.6f}")
    print("\n3. 最右侧所有 RX 门应该填入的值：")
    print(f"  所有 RX 门统一填写 -> {2 * solution[0]:.6f}")
    print("\n----------------- Full result ---------------------", flush=True)
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------", flush=True)
    
# 1. 在循环前定义一个用来保存状态的列表
    value_save = []
    probability_save = []
    selection_save = []  # <--- 新增这行
    
    for i in range(len(result_sorted)):
        x, probability = result_sorted[i]
        value = np.real(value_mm[i])
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x[::-1], value, probability), flush=True)
        
        value_save.append(value)
        probability_save.append(probability)
        selection_save.append(x[::-1])  # <--- 新增这行，保存翻转对齐后的真实选股状态（如 110101）
        
    os.makedirs("./output", exist_ok=True)
    # 2. 在 np.savez 中把 selection 也作为关键词存进去
    np.savez("./output/budget_{}_layers_{}_eta_{}.npz".format(budget, layers, eta), 
             value=np.array(value_save), 
             probability=np.array(probability_save),
             selection=np.array(selection_save)) # <--- 修改这行
    print(f"\n结果数据已成功保存至 ./output/budget_{budget}_layers_{layers}_eta_{eta}.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=4, help='Total assets.')
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')
    parser.add_argument('--g', type=int, default=1, help='Binary bits per asset.')
    parser.add_argument('--half_q', type=float, default=0.25, help='Coefficient of the quadratic term.')
    parser.add_argument('--eta', type=float, default=6.0, help='Coefficient of the Lagrangian term.')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed.')
    parser.add_argument('--optimizer', action='store_true', default=False, help='Use SciPy COBYLA instead of Qiskit SPSA.')
    parser.add_argument('--maxiter', type=int, default=500, help='Max iterations.')
    parser.add_argument('--layers', type=int, default=4, help='The number of QAOA layers.')
    args = parser.parse_args()

    budget = args.budget
    half_q = args.half_q
    eta = args.eta
    num_assets = args.num_assets
    num_slices = args.g  
    layers = args.layers
    num_qubits = num_assets * num_slices

    print_config()
    np.random.seed(args.seed)

    # ==================== 恢复动态计算数据功能 ====================
    file_path = "./data/stock_data.xlsx"
    print(f"\n正在从 {file_path} 读取股票收盘价并动态计算均值和协方差...")
    try:
        exp_ret, cov_mat = data_preprocessing(file_path)
    except Exception as e:
        print(f"❌ 读取或计算失败，请确保表格路径正确且格式规范。错误原因: {e}")
        exit(1)

    print('\n%%%%%%%%%%%%%%%%%%%% Calculated Input Data %%%%%%%%%%%%%%%%%%%%')
    print("Expected Returns (mu):\n", exp_ret)
    print("-" * 50)
    print("Covariance Matrix (Sigma):")
    with np.printoptions(precision=6, suppress=True):
        print(cov_mat)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

    J = calc_J(num_qubits, half_q, cov_mat, eta)
    h = calc_h(num_qubits, num_assets, half_q, cov_mat, exp_ret, eta, budget)
    Pauli_sum = problem_PauliOperator(num_qubits, h, J)
    
    simulator = AerSimulator(method='statevector', device='CPU')
    print("⚠️ 未检测到 GPU 环境或未配置 CUDA，已自动切回 CPU 单线程运行。")
    simulator.set_options(max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1, statevector_parallel_threshold=14)
    
    # 建立参数化线路
    qc = QuantumCircuit(num_qubits)
    beta = [Parameter(f"β{i}") for i in range(layers)]
    gamma = [Parameter(f"γ{i}") for i in range(layers)]
    para_list = beta + gamma

    qc.append(insert_H(num_qubits), range(num_qubits))
    for i in range(layers):
        qc.append(oneCircuit(num_qubits, h, J, beta[i], gamma[i]), range(num_qubits))
    qc.save_statevector()
    
    qc_compiled = transpile(qc, simulator)
    print('Circuit Initialization Complete! Start Training...\n')

    expectation = get_expectation(qc_compiled, para_list, Pauli_sum, num_qubits, simulator)

    start = time.time()
    init_point = np.random.uniform(0, 0.001 * np.pi, size=layers * 2)
    
    if args.optimizer:
        from scipy.optimize import minimize
        print("Using Scipy COBYLA optimizer...")
        res = minimize(expectation, init_point, method='COBYLA', options={'maxiter': args.maxiter})
        solution = res.x
    else:
        print("Using Qiskit SPSA optimizer...")
        callback_func = callback(step_size=1)
        optimizer = SPSA(maxiter=args.maxiter, blocking=True, second_order=True, callback=callback_func)
        res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=init_point)
        solution = res[0]

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time() - start))

    print_result(qc, Pauli_sum.to_matrix(), para_list, solution, num_qubits, simulator, budget, layers, eta)

    # 导出可视化线路图
    print("\n正在生成量子线路架构图...")
    try:
        qc_draw = QuantumCircuit(num_qubits)
        qc_draw.append(insert_H(num_qubits), range(num_qubits))
        for i in range(layers):
            qc_draw.append(oneCircuit(num_qubits, h, J, solution[i], solution[i+layers]), range(num_qubits))
        
        fig = qc_draw.decompose().draw(output='mpl', style='iqp')
        fig.savefig('./output/qaoa_circuit.png', bbox_inches='tight')
        print("线路展开图已成功导出至: ./output/qaoa_circuit.png")
    except Exception as e:
        print(f"线路图导出失败: {e}")