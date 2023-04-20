from utils import *
from copy import deepcopy
from scipy.stats import linregress


filenames = {} 
with st.sidebar:
    tab0, tab1, tab2 = st.tabs(['调优参数', '通用参数', '数据文件'])
    with tab0:
        col0, col1 = st.columns(2)
        with col0:
            dx = st.number_input("x偏移量(r/min)", min_value=-100, max_value=100, value=0, step=1) 
            θ1 = st.number_input("角度1", min_value=0., max_value=1.2, value=.5, step=0.01) 
            h = st.number_input("限扭", min_value=800, max_value=2000, value=1300, step=1) 

        with col1:
            dy = st.number_input("y偏移量(N*m)", min_value=-100, max_value=100, value=0, step=1) 
            θ2 = st.number_input("角度2", min_value=0., max_value=.25, value=.2, step=.005) 
            cp = st.number_input("控制点", min_value=10, max_value=200, value=30, step=1) 

        engine_ = st.file_uploader("发动机数据(优化后)(N, Me)", type="xlsx")

    with tab1:
        col0, col1 = st.columns(2)
        with col0:
            MPTO = st.number_input("液压系统扭矩(N*m)", min_value=0, max_value=1000, value=85, step=1)
            im = st.number_input("总速比", min_value=1., max_value=100., value=56.17, step=1., format="%.2f")
            φ = st.number_input("附着系数", min_value=0., max_value=1., value=.75, step=.01) # 0-1
            Gs = st.number_input("装载机重量(kg)", min_value=1, max_value=50000, value=17100, step=1) # 
            δ0 = st.number_input("旋转质量转换系数", min_value=1., max_value=1.3, value=1.08, step=.01) # 1.08
            g = st.number_input("重力加速度(m/s^2)", min_value=9.5, max_value=10., value=9.8, step=.01) # 1.08

        with col1:
            n_i_split = st.number_input("传动比切分点", min_value=100, max_value=500, value=100, step=1)
            ηm = st.number_input("机械总效率", min_value=0., max_value=1., value=.8, step=.01) # 0-1
            rk = st.number_input("轮胎滚动半径(m)", min_value=.001, max_value=2., value=.725, step=.001, format="%.3f") # .75m
            α = st.number_input("坡度(°)", min_value=-90, max_value=90, value=0, step=1) # 
            f = st.number_input("滚动阻力系数", min_value=0., max_value=1., value=.03, step=.01) 
            Hz = st.number_input("作业采集频率", min_value=1, max_value=100, value=50, step=1) 

    with tab2:
            filenames['engine'] = st.file_uploader("发动机数据(N, Me, Mba)", type="xlsx", help='')
            filenames['converter'] = st.file_uploader("变矩器(i, N, Mb, K)", type="xlsx")
            filenames['cons'] = st.file_uploader("万有数据(N, Me, ge, P)", type="xlsx")
            filenames['smoking'] = st.file_uploader("烟线(N, Me)", type="xlsx")
            filenames['operation'] = st.file_uploader("作业数据(N, Me)", type="xlsx")

data = loaddata(filenames)
model, SL, INT = calculate(data, MPTO, n_i_split, im, ηm, φ, rk, Gs, α, δ0, f, g, Hz)

tabs = st.tabs(["曲线调整", "外特性", '速度特性', "动力匹配", "交值数据", "整机特性", "车速分析"])

data_ = deepcopy(data)
if not engine_: 
    with tabs[0]:
        Neh = 2000
        Meh = model['Me'](Neh)
        Fx = Neh + dx
        Fy = Meh + dy
        st.markdown('### 外特性曲线调优')
        fig, ax = plot('转速(r/min)', '扭矩(N*m)', plist=[
            [[SL['x_N'], SL['y_Me']], {'label':'原始'}],
            [[SL['x_N'], SL['y_smk']], {'label':'烟线'}],
            [[data['engine']['N'], data['engine']['Me'], 'o'], {}],
            [[Neh, Meh, 'o'], {'label':'额定点', 'color': 'orange'}],
            [[Fx, Fy, 'o'], {'label':'优化中心', 'color': 'red'}],
            [[[SL['x_N'].iloc[0], Fx], [Meh, Meh], '--'], {'label': '基准线', 'color': 'grey'}],
            [[[Fx, Fx], [0, Meh], '--'], {'color': 'grey'}],
            [[[SL['x_N'].iloc[0], Fx], [np.tan(θ1) * Fx + Fy , Meh], 'r--'], {'label': '调优线'}],
            [[[Fx, np.tan(θ2) * Fy + Fx], [Meh, 0], 'r--'], {}],
            [[SL['x_N'], [h] * len(SL), '--'], {'label':'限扭', 'color': 'orange'}],
        ], show=False)
        ax.grid(linestyle = '--', linewidth = 0.5)
        ax.legend(loc='lower left')
        st.pyplot(fig)

        model['Me_'] = interp1d(
            [SL['x_N'].iloc[0], Fx, np.tan(θ2) * Fy + Fx],
            [np.tan(θ1) * Fx + Fy, Fy, 0],
            kind='linear', bounds_error=False
        ) 

        y_Me_ = model['Me_'](SL['x_N'])
        y_Me_[ y_Me_ > h] = h
        y_Me_[ y_Me_ > SL['y_smk']] = SL['y_smk'][ y_Me_ > SL['y_smk']]
        y_Me_ = pd.Series(y_Me_).fillna(0)
        # st.write(y_Me_)
        model['Me_'] = interp1d(SL['x_N'], y_Me_, kind='cubic', bounds_error=False) 

        data_['engine'] = pd.DataFrame()
        data_['engine']['N'] = list(np.linspace(data['engine']['N'].min(), data['engine']['N'].max(), cp))
        data_['engine']['Me'] = model['Me_'](data_['engine']['N'])
        data_['engine']['Mba'] = model['Mba'](data_['engine']['N'])
else:
    data_['engine'] = pd.read_excel(engine_)
    with tabs[0]:
        st.success('优化数据为外部导入')


model_, SL_, INT_ = calculate(data_, MPTO, n_i_split, im, ηm, φ, rk, Gs, α, δ0, f, g, Hz)

with tabs[1]:
    st.markdown('### 外特性曲线对比')
    st.download_button(label="下载优化后数据", 
        data=dataframe_to_bytes(data_['engine']),file_name='发动机外特性(优化后).xlsx', mime='xls/xlsx')

    data['operation']['g'] = ((data['operation']['N'] / 20).round() * 20)
    value_counts = data['operation']['g'].value_counts()
    fig, ax = arrow_figure()
    ax.plot(SL['x_N'], SL['y_Me'], label='原始')
    ax.plot(data['engine']['N'], data['engine']['Me'], 'bo', markersize=3)
    ax.plot(SL_['x_N'], SL_['y_Me'], 'r-', label='优化后')
    ax.plot(data_['engine']['N'], data_['engine']['Me'], 'bo', label='控制点', markersize=3)
    ax.grid()
    ax2 = add_secondary_axes(fig)
    ax2.bar(list(value_counts.index), list(value_counts), width=10, label='作业数据', color='orange', alpha=0.5)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    st.pyplot(fig)
    

with tabs[2]:
    st.markdown('### 发动机速度特性曲线')
    fig, ax = arrow_figure()
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.2))

    ax.plot(data['engine']['N'], data['engine']['Me'], "o", label="实验值")
    p1, = ax.plot(SL['x_N'], SL['y_Me'], color='b', label="仿真值")

    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_xlabel("转速(r/min)")
    ax.set_ylabel("扭矩(N*m)")
    ax.legend()

    p2, = ax2.plot(SL['x_N'], SL['y_ge'], color="r", label='比油耗(g/kw*h)')
    ax2.set_ylabel('比油耗(g/kw*h)')

    p3, = ax3.plot(SL['x_N'], SL['y_Pb'], color="g", label='功率(kw/h)')
    ax3.set_ylabel('功率(kw/h)')

    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    st.pyplot(fig)

with tabs[3]:
    st.markdown('### 动力匹配图')
    plot('转速(r/min)', '扭矩(N*m)', plist=\
        [[[SL['x_N'], y_Mb], {'color': 'grey', 'linewidth': 1}] \
            for _, y_Mb in SL.filter(regex=r'y_Mb\[\d+\]').items()] + [
            [[SL['x_N'], SL['y_Mb']], {'label': '原始'}],
            [[SL_['x_N'], SL_['y_Mb']], {'label': '优化后'}],
            [[INT['N'], INT['Me'], 'o'], {'color': 'blue', 'markersize': 1}],
            [[INT_['N'], INT_['Me'], 'o'], {'color': 'red', 'markersize': 1}]
        ]
    )

with tabs[4]:
    st.markdown('### 匹配交值表')
    tab1, tab2 = st.tabs(["优化后", "原始"])
    with tab1:
        INTV0_ = INT_[meas]
        INTV0_= INTV0_.rename(columns={
            code: f"{ name } / {code} ({ unit })" for code, (name, unit) in meas.items()
        })
        st.download_button(label="下载交值表", 
            data=dataframe_to_bytes(INTV0_),file_name='交值表.xlsx', mime='xls/xlsx')
        st.dataframe(INTV0_.T.style.format("{:.2f}"), height=900)
    with tab2:
        INTV0 = INT[meas]
        INTV0= INTV0.rename(columns={
            code: f"{ name } / {code} ({ unit })" for code, (name, unit) in meas.items()
        })
        st.download_button(label="下载交值表", 
            data=dataframe_to_bytes(INTV0),file_name='交值表.xlsx', mime='xls/xlsx')
        st.dataframe(INTV0.T.style.format("{:.2f}"), height=900)


with tabs[5]:
    st.markdown('### 交值图表') 
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(label='x-轴', options=INTV0.columns, index=0)
    with col2:
        # y_axis = st.selectbox(label='y-轴', options=INTV0.columns, index=1)
        Y_axis = st.multiselect(label='y-轴', options=INTV0.columns, default=['加速度 / a (m/s^2)'])

    if Y_axis:
        INTV0 = INTV0.sort_values(x_axis)
        INTV0_ = INTV0_.sort_values(x_axis)
        fig, ax = arrow_figure()
        ax.set_xlabel(x_axis)
        ax.grid()
        labels = ['原始', '优化后'] 
        for i, y_axis in enumerate(Y_axis):
            if i != 0:
                ax = add_secondary_axes(fig)
                labels = [None, None]
            ax.set_ylabel(y_axis)
            ax.plot(INTV0[x_axis], INTV0[y_axis], label=labels[0], color=colors[i])
            ax.plot(INTV0_[x_axis], INTV0_[y_axis], '--', label=labels[1], color=colors[i])
            # ax.yaxis.label.set_color(colors[i])
        legend_shift = i * 0.05
        fig.legend(loc='upper center', bbox_to_anchor=(0.5 + legend_shift, -0.05), fancybox=True, shadow=True, ncol=5)
        st.pyplot(fig)

with tabs[6]:
    st.markdown('### 车速分析') 
    sr = st.slider("速度范围(km/h)", min_value=50, max_value=200, value=60, step=1) 
    n_points = int(len(INT) * 5. / 100.)

    Ff = INT['Ff'][0]
    reg = linregress(INT['Vt'][-n_points:], INT['Fk'][-n_points:])
    x_int = -reg.intercept / reg.slope
    vm = (Ff-reg.intercept) / reg.slope

    Ff_ = INT_['Ff'][0]
    reg_ = linregress(INT_['Vt'][-n_points:], INT_['Fk'][-n_points:])
    x_int_ = -reg_.intercept / reg_.slope
    vm_ = (Ff_-reg_.intercept) / reg_.slope

    tab1, tab2 = st.tabs(["优化后", "原始"])
    with tab1:
        fig, ax = plot('理论行驶速度 Vt (km/h)', '切线牵引力 / Fk (KN)', plist=[
            [[INT_['Vt'], INT_['Fk']], {'label':'切线牵引力'}],
            [[range(sr), [Ff_] * sr, '--'], {'label':'滚动阻力'}],
            [[[INT_['Vt'].iloc[-1], x_int_], [INT_['Fk'].iloc[-1], 0], '--'], {'label':'延长线'}],
            [[vm_,  Ff_, 'o'], {'label':'交点'}],
        ], show=False)
        ax.annotate(
            f'最大车速: {vm_:.1f}km/h', xy=(vm_+1, Ff+1), xytext=(vm_+5, Ff_+10),
            arrowprops=dict(facecolor='black', shrink=0.01))
        ax.grid(linestyle = '--', linewidth = 0.5)
        ax.legend()
        st.pyplot(fig)

    with tab2:
        fig, ax = plot('理论行驶速度 Vt (km/h)', '切线牵引力 / Fk (KN)', plist=[
            [[INT['Vt'], INT['Fk']], {'label':'切线牵引力'}],
            [[range(sr), [Ff] * sr, '--'], {'label':'滚动阻力'}],
            [[[INT['Vt'].iloc[-1], x_int], [INT['Fk'].iloc[-1], 0], '--'], {'label':'延长线'}],
            [[vm,  Ff, 'o'], {'label':'交点'}],
        ], show=False)
        ax.annotate(
            f'最大车速: {vm:.1f}km/h', xy=(vm+1, Ff+1), xytext=(vm+5, Ff+10),
            arrowprops=dict(facecolor='black', shrink=0.01))
        ax.grid(linestyle = '--', linewidth = 0.5)
        ax.legend()
        st.pyplot(fig)
