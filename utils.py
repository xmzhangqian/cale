import io

import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
matplotlib.rc('font', family='Microsoft YaHei', size=20) 
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', ':', '-.']

def dataframe_to_bytes(df):
    """  """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.close()
    output.seek(0)
    return output.read()

def intersect(y1, y2):
    return np.argwhere(np.diff(np.sign(y1 - y2))).flatten() + 1

def arrow_figure(figsize=(6, 4)):
    """ """
    fig = plt.figure(figsize=figsize)
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis['right'].set_visible(False)
    ax.axis['top'].set_visible(False)
    ax.axis['bottom'].set_axisline_style("-|>", size = 1.0)
    ax.axis['left'].set_axisline_style("-|>", size = 1.0)
    return fig, ax

def add_secondary_axes(fig):
    """ """
    axes = fig.get_axes()
    ax = axes[0]
    ax2 = ax.twinx()
    n = len(axes) - 1
    if n > 0:
        ax2.spines.right.set_position(("axes", 1 + .2 * n))
    return ax2

def plot(xlabel, ylabel, plist, show=True):
    """  
    plist = [item0, item1, ...]
    item = [args, kwargs] 
    """
    fig, ax = arrow_figure()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for p in plist: plt.plot(*p[0], **p[1])
    if show: 
        ax.grid(linestyle = '--', linewidth = 0.5)
        ax.legend()
        st.pyplot(fig)
    return fig, ax

def pd2np(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    return x

@st.cache_resource
def calculate(data, MPTO, n_i_split, im, ηm, φ, rk, Gs, α, δ0, f, g, Hz):
    """ """
    model = {}
    model['Me'] = interp1d(data['engine']['N'], data['engine']['Me'], kind='cubic', bounds_error=False) 
    model['Mba'] = interp1d(data['engine']['N'], data['engine']['Mba'], kind='cubic', bounds_error=False)
    model['Mb1k'] = interp1d(data['converter']['i'], data['converter']['Mb'], kind='cubic', bounds_error=False)
    model['Mb'] = lambda N, i: (N / 1000) ** 2 * model['Mb1k'](i)
    model['K'] = interp1d(data['converter']['i'], data['converter']['K'], kind='cubic', bounds_error=False)
    model['ge'] = RegularGridInterpolator(
        (data['consmap'].T.index.values, data['consmap'].T.columns.values), 
        data['consmap'].T.values, method='cubic', bounds_error=False)
    model['smk'] = interp1d(data['smoking']['N'], data['smoking']['Me'], 
        kind='cubic', bounds_error=False, fill_value=data['smoking']['Me'].min())

    SL = pd.DataFrame()
    SL['x_N'] = np.linspace(data['engine']['N'].min(), data['engine']['N'].max(), 10000)
    x_i = np.linspace(data['converter']['i'].min(), data['converter']['i'].max(), n_i_split)
    Y_Mb = list(map(lambda i: model['Mb'](SL['x_N'], i), x_i))
    SL['y_Me'] = model['Me'](SL['x_N'])
    SL['y_Me'][SL['y_Me'] < 0] = 0
    SL['y_Mba'] = model['Mba'](SL['x_N'])
    SL['y_Mb'] = SL['y_Me'] - SL['y_Mba'] - MPTO
    SL['y_Pb'] = SL['x_N'] * SL['y_Me'] / 9550
    SL['y_ge'] = model['ge'](np.array([SL['x_N'], SL['y_Me']]).T)
    SL['y_smk'] = model['smk'](SL['x_N'])
    for i, y_Mb in enumerate(Y_Mb): SL[f'y_Mb[{i}]'] = y_Mb

    # 计算交点
    xi = np.array(list(map(lambda y_Mb: intersect(SL['y_Mb'], y_Mb)[0], Y_Mb)))

    INT = pd.DataFrame()
    INT['i'] = x_i
    INT['N'] = SL['x_N'][xi].to_list()
    INT['Me'] = SL['y_Mb'][xi].to_list()
    INT['Mb1k'] = list(map(lambda i: model['Mb'](1000, i), x_i))
    # K值
    INT['K'] = model['K'](x_i)
    # 扭矩(输出) Mt
    INT['Mt'] = INT['Me'] * INT['K']
    # 转速(输出) Nt
    INT['Nt'] = INT['N'] * x_i
    # 功率       Pb
    INT['Pb'] = INT['Me'] * INT['N'] / 9550
    # 功率(输出) Pt
    INT['Pt'] = INT['Mt'] * INT['Nt'] / 9550
    # 传动效率(%) η 
    INT['η'] = INT['Pt'] / INT['Pb'] * 100
    # 高效区门限
    INT['ηe'] = 75
    # 功率损失 = l
    INT['l'] = INT['Pb'] - INT['Pt']
    # 比油耗 ge
    INT['ge'] = model['ge'](np.array([INT['N'], INT['Me']]).T) 
    # 油耗 = Ge (有乘与功率的为G)
    INT['Ge'] = INT['ge'] * INT['Pb']
    # 切线牵引力 Fk 
    INT['Fk'] = INT['Mt'] * im * ηm / rk / 1000 
    # 滚动阻力 Ff
    INT['Ff'] = Gs * g * .03 * np.cos(α) / 1000
    # 坡道阻力 Fi
    INT['Fi'] = Gs * g * .03 * np.sin(α) / 1000
    # 有效牵引力 Fkp
    INT['Fkp'] = INT['Fk'] - INT['Ff'] - INT['Fi']
    # 最大有效牵引力 Fφ 
    INT['Fφ'] = Gs * φ 
    # 理论行驶速度 vt 
    INT['Vt'] = .377 * rk * INT['Nt'] / im
    # 滑转率 δ
    INT['δ'] = .1 * INT['Fkp'] / Gs + 9.25 * (INT['Fkp'] / Gs) ** 8 
    # 实际行驶速度 v
    INT['v'] = INT['Vt'] * (1 - INT['δ'])
    # 牵引功率 Pkp
    INT['Pkp'] = INT['Fkp'] * INT['v']
    # 额定功率 Peh
    INT['Peh'] = model['Me'](2000) * 2000 / 9550
    # 牵引效率 ηkp
    INT['ηkp'] = INT['Pkp'] / INT['Peh']
    # 装载机等效质量 mveh
    INT['mveh'] = Gs * δ0 
    # 加速度 a
    INT['a'] = INT['Fkp'] / INT['mveh'] * 1000
    # Δ时间 Δt
    INT['Δt'] = INT['v'].diff().fillna(0) / INT['a'].rolling(2).mean().fillna(INT['a'])
    # t时间 t
    INT['t'] = INT['Δt'].cumsum()
    # Δ位移 Δs
    INT['Δs'] = INT['v'].rolling(2).mean().fillna(INT['v']) * INT['Δt']
    # s位移 s
    INT['s'] = INT['Δs'].cumsum()

    # 作业总油耗 
    OP = pd.DataFrame()
    OP['N'] = data['operation']['N'].apply(lambda x: 800 if x < 800 else x)
    OP = OP.query(f"{INT['N'].min()} <= N <= {INT['N'].max()}")

    INT['OPc'] = pd.Series(np.abs(
        OP['N'].values.reshape((len(OP), 1)) - INT['N'].values.reshape((1, len(INT)))
    ).argmin(axis=1)).value_counts().sort_index() 
    INT['OPc'] = INT['OPc'].fillna(0)
    INT['OPt'] = INT['OPc'] * (1. / Hz)

    # 平均油耗 MGe
    OP = pd.DataFrame()
    OP['N'] = data['operation']['N'].apply(lambda x: 800 if x < 800 else x)
    OP['Me'] = model['Me'](OP['N'])
    OP = OP.dropna()
    
    OP['ge'] = model['ge'](np.array([OP['N'], OP['Me']]).T) 
    OP['Pb'] = OP['Me'] * OP['N'] / 9550
    OP['Ge'] = OP['ge'] * OP['Pb']
    INT['MGe'] = (OP['ge'] * OP['Pb']).mean()

    return model, SL ,INT

meas = {
    'i': ['传动比', '/'],
    'N': ['转速(输入)', 'r/min'],
    'Me': ['扭矩(输入)', 'N*m'],
    'K': ['K值', '/'],
    'Mt': ['扭矩(输出)', 'N*m'],
    'Nt': ['转速(输出)', 'r/min'],
    'Pb': ['功率(输入)', 'kw'],
    'Pt': ['功率(输出)', 'kw'],
    'η': ['传动效率', '%'],
    'l': ['功率损失', 'W'],
    'ge': ['比油耗', 'g/kw*h'],
    'Ge': ['油耗', 'g/h'],
    'MGe': ['平均油耗', 'g/h'],
    'Fk': ['切线牵引力', 'KN'],
    'Ff': ['滚动阻力', 'KN'],
    'Fi': ['坡道阻力', 'KN'],
    'Fkp': ['有效牵引力', 'KN'],
    'Fφ': ['最大有效牵引力', 'KN'],
    'Vt': ['理论行驶速度', 'km/h'],
    'δ': ['滑转率', '/'],
    'v': ['实际行驶速度', 'km/h'],
    'Pkp': ['牵引功率', 'kw'],
    'Peh': ['额定功率', 'kw'],
    'ηkp': ['牵引效率', '%'],
    'a': ['加速度', 'm/s^2'],
    'Δt': ['Δ时间', 's'],
    't': ['时间', 's'],
    'Δs': ['Δ位移', 'm'],
    's': ['位移', 'm'],
    'OPc': ['采集点', 'n'],
    'OPt': ['工作时间', 't'],
}

def loaddata(filenames):
    data = {}
    for name, content in filenames.items():
        xlsx = content if filenames[name] else f'data/{ name }.xlsx'
        data[name] = pd.read_excel(xlsx)
    data['consmap'] = data['cons'].copy()
    data['consmap']['Me'] = data['consmap']['Me'].apply(lambda x: round(x / 100) * 100)
    data['consmap']['N'] = data['consmap']['N'].apply(lambda x: round(x / 100) * 100)
    data['consmap'] = data['consmap'].groupby(['N','Me'])['ge'].mean().unstack('N')
    data['consmap'] = data['consmap'].fillna(method='pad')
    data['smoking']['N'] = data['smoking']['N'].apply(lambda x: round(x / 100) * 100)
    data['smoking'] = data['smoking'].groupby('N')['Me'].min().reset_index()
    data['operation'] = data['operation'].query('N > 800 and Me > 0')
    return data