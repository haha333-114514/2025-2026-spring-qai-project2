import argparse
import time
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
import pandas as pd


def data_preprocessing(file_path):
    '''
    返回期望收益向量和协方差矩阵
    '''
    ## excel --> array
    df = pd.read_excel(file_path)
    data = df.values
    ## closing price --> rate of return
    RoR = np.zeros([data.shape[0]-1,data.shape[1]])   # Rate of return
    for i in range(data.shape[0]-1):
        RoR[i,:] = (data[i+1,:]-data[i,:])/data[i,:]
    ## 计算期望收益和协方差


    '''
    code here
    '''



    return exp_ret, cov_mat


def calc_J():
    '''
    计算Rzz门系数，返回一个系数矩阵
    '''
    J = np.zeros((num_qubits, num_qubits))

    '''
    code here
    '''



    return J

def calc_h():
    '''
    计算Rz门系数，返回一个系数向量
    '''
    h = np.zeros(num_qubits)

    '''
    code here
    '''

    return h

def insert_RX():
    '''
    对量子线路qc中所有qubit添加Rx门，返回添加完的qc
    '''
    qc = QuantumCircuit(num_qubits)


    '''
    code here
    '''


    return qc

def insert_RZ():
    '''
    对量子线路qc中所有qubit添加Rz门，返回添加完的qc
    '''
    qc = QuantumCircuit(num_qubits)


    '''
    code here
    '''


    return qc

def insert_RZZ(gamma, J):
    '''
    对量子线路qc中所有qubit对添加Rzz门，Rzz门系数为返回添加完的qc
    注：（1，3）与（3，1）视为同一个qubit对，即每两个qubit间只需要添加一次Rzz门即可
    '''
    qc = QuantumCircuit(num_qubits)

    '''
    code here
    '''


    qc.barrier()
    return qc

def insert_H():
    '''
    对量子线路qc中所有qubit添加Hadamard门，返回添加完的qc
    '''
    qc = QuantumCircuit(num_qubits)


    '''
    code here
    '''



    return qc


#接下来的两个函数拼接出任务对应的哈密顿量（qiskit允许使用门电路表达形式）
def get_Pauli(index, type):
    if type == 'Z':
        assert len(index) == 1
        index = index[0]
        assert index >= 0 and index <= num_qubits - 1
        _Pauli = ['I'] * (num_qubits - 1)
        _Pauli.insert(index, 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    elif type == 'ZZ':
        assert len(index) == 2
        _Pauli = ['I'] * (num_qubits - 2)
        for i in range(len(index)):
            assert index[i] >= 0 and index[i] <= num_qubits - 1
            _Pauli.insert(index[i], 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    else:
        raise AssertionError()

def problem_PauliOperator(h, J):
    Pauli_h_list = []
    for i in range(num_qubits):
        Pauli_h_list.append((get_Pauli([i], 'Z'), h[i]))
    Pauli_h = PauliSumOp.from_list(Pauli_h_list, coeff=1.0)

    Pauli_J_list = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli_J_list.append((get_Pauli([i,j], 'ZZ'), J[i][j]))
    Pauli_J = PauliSumOp.from_list(Pauli_J_list, coeff=1.0)

    Pauli_sum = Pauli_h + Pauli_J

    return Pauli_h, Pauli_J, Pauli_sum



def oneCircuit(h, J, beta, gamma):
    '''
    拼接出一层中间层，一层中间层由两层组成，一层为混合层，一层为损失层
    '''
    #提示，使用qc.append()
    qc = QuantumCircuit(num_qubits)

    '''
    code here
    '''


    return qc

def get_expectation(circuit, para_list, Hamiltonian):

    def execute_circ(theta):
        #theta为线路中所有可变参数，前一半为beta，后一半为gamma
        qc = QuantumCircuit(num_qubits)

        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i+p]] = gamma[i]
        
        qc.append(circuit, [i for i in range(0, num_qubits)])
        qc.assign_parameters(para_dict, inplace=True)
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        _statevector = result.get_statevector(circ)

        #利用当前状态向量计算loss
        loss = _statevector.expectation_value(Hamiltonian)


        assert np.imag(loss) < 1e-10
        return np.real(loss)

    return execute_circ

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

def print_config():
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, eta: %f, layers: %d' % (budget, num_slices, eta, layers))

def print_result(circuit, Hamiltonian, para_list, solution):
    qc = QuantumCircuit(num_qubits)

    p = len(solution) // 2
    beta = solution[:p]
    gamma = solution[p:]

    para_dict = {}
    for i in range(p):
        para_dict[para_list[i]] = beta[i]
        para_dict[para_list[i + p]] = gamma[i]
    qc.append(circuit, [i for i in range(0, num_qubits)])
    qc.assign_parameters(para_dict, inplace=True)
    circ = transpile(qc, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ)  # innner product of statevector_dagger and statevector is 1
    statevector = statevector.to_dict()
    a = 0
    for i in statevector:
        statevector[i] = np.abs(np.array(statevector[i])) ** 2
        a = a + statevector[i]
    # print('a: %f' % a)
    # print(statevector)
    result = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # print(result)
    mm = []
    for i in range(len(result)):
        x, _ = result[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)
    value_mm = np.sum((mm @ Hamiltonian) * mm, axis=1)

    min_index = np.argmin(value_mm)
    print("\nOptimal: selection {}, value {:.8f}".format(result[min_index][0][::-1], value_mm[min_index]))

    print("\n----------------- Full result ---------------------", flush=True)
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------", flush=True)
    value_save = []
    probability_save = []
    utility_save = []
    for i in range(len(result)):
        x, probability = result[i]
        value = value_mm[i]
        assert np.imag(value) < 1e-10
        value = np.real(value)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x[::-1], value, probability), flush=True)
        ## do not save the optimal selection
        np.savez("./output/budget_{}_layers_{}_eta_{}.npz".format(budget, layers, eta), value=np.array(value_save), \
        probability=np.array(probability_save), utility=np.array(utility_save))

class callback:
    def __init__(self, step_size: int):
        self.step_size = step_size
        self.full_values = []
        self._values = []
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.full_values.append(value)
        self._values.append(value)
        if len(self._values) == self.step_size:
            last_value = self._values[-1]
            self.values.append(last_value)
            self._values = []
            return self.values

def print_loss(res):
    print('%%%%%%%%%%%%%%%%%%%% Optimization Output %%%%%%%%%%%%%%%%%%%%')
    loss_ls = callback_func.values
    print('minimal loss: %s, \nmaxIter: %d, func_eval: %d' % (res[1], len(callback_func.full_values), res[2]))
    print("Parameters Found:", res[0])
    print("\n----------------- Loss (%d steps from %d iterations) -----------------" % (len(loss_ls), len(callback_func.full_values)))
    print("iter\t\tloss")
    print("------------------------------------------------------------------------")
    for i in range(len(loss_ls)):
        loss = loss_ls[i]
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t\t%.10f" % (i, loss))


if __name__ == '__main__':
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=4, help='Total assets.')
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')
    parser.add_argument('--g', type=int, default=1, help='The number of binary bits required to represent one asset.')
    parser.add_argument('--theta1', type=float, default=1.0, help='Coefficient of the linear term.')
    parser.add_argument('--half_q', type=float, default=0.25, help='Coefficient of the quadratic term.')
    parser.add_argument('--eta', type=float, default=1.0, help='Coefficient of the Lagrangian term.')
    parser.add_argument('--seed', type=int, default=123456, help='Randon seed.')
    parser.add_argument('--optimizer', action='store_true', default=False, help='use scipy optimizer.')
    parser.add_argument('--maxiter', type=int, default=300, help='max iterations.')
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--layers', type=int, default=3, help='The number of QAOA layers.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    args = parser.parse_args()

    # 初始化参数
    budget = args.budget
    Gf = 1.0 / budget
    theta1 = 1.0
    half_q = 0.25
    eta = args.eta
    num_assets = args.num_assets
    num_slices = args.g  # The number of binary bits required to represent one asset
    layers = args.layers

    print_config()

    num_qubits = num_assets * num_slices

    np.random.seed(args.seed)
    # 读取收益和方差
    file_path = "./data/stock_data.xlsx"
    exp_ret, cov_mat = data_preprocessing(file_path)
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()

    # 计算所给问题对应的哈密尔顿量的系数
    J = calc_J()
    h = calc_h()

    # 计算所给问题对应的哈密尔顿量
    Pauli_h, Pauli_J, Pauli_sum = problem_PauliOperator(h, J)
    
    # 初始化量子虚拟机, 分配量子比特
    simulator = AerSimulator(method = 'statevector')

    # 比特数较大时可以调整这里的参数进行并行计算，本次任务线路较小不需要修改。
    simulator.set_options(
        # default parameters
    max_parallel_threads = 0,
    max_parallel_experiments = 1,
    max_parallel_shots = 0,
    statevector_parallel_threshold = 14
)
    
    qc = QuantumCircuit(num_qubits)

    # 配置待优化参数
    beta = []
    gamma = []
    para_list = []
    for i in range(layers):
        name = "β%d" % i
        beta.append(Parameter(name))
        name = "γ%d" % i
        gamma.append(Parameter(name))

    para_list = beta + gamma

    # 构建QAOA
    qc.append(insert_H(), [i for i in range(0, num_qubits)])
    ##  
    for i in range(layers):
        qc.append(oneCircuit(h, J, beta[i], gamma[i]), [i for i in range(0, num_qubits)])
    qc.save_statevector()
    print('\nCircuit Initialization Complete! Start Training...', flush=True)


    
    # 计算loss
    expectation = get_expectation(qc, para_list, Pauli_sum)

    # 优化参数
    start = time.time()
    if args.optimizer:
        # 利用外部优化器
        res = minimize(expectation,
                       np.random.uniform(0, 0.001 * np.pi, size=layers * 2),
                       method='COBYLA',
                       options={'maxiter': args.maxiter})
        print('\nTraining Done! The output of optimizer: ')
        print(res)
        solution = res.x
    else:
        # 利用qiskit自带优化器
        # optimizer = COBYLA(maxiter=args.maxiter, tol=0.0001)
        # res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, np.pi, size=layers * 2))
        step_size = 1 # 每隔step_size个iterations打印一次loss
        callback_func = callback(step_size)
        optimizer = SPSA(maxiter=args.maxiter, blocking=True, second_order=True, callback=callback_func)
        res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, 0.001*np.pi, size=layers * 2))
        solution = res[0]
        # 打印loss的变化
        print_loss(res)

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time()-start))

    # 打印结果
    print_result(qc, Pauli_sum.to_matrix(), para_list, solution)


