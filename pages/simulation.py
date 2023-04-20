from utils import *
from scipy.stats import linregress

filenames = {} 
with st.sidebar:
    tab0, tab1 = st.tabs(['参数', '数据文件'])
    with tab0:
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
            Hz = st.number_input("作业采集频率(Hz)", min_value=1, max_value=100, value=50, step=1) 

    with tab1:
        filenames['engine'] = st.file_uploader("发动机数据(N, Me, Mba)", type="xlsx")
        filenames['converter'] = st.file_uploader("变矩器(i, N, Mb, K)", type="xlsx")
        filenames['cons'] = st.file_uploader("万有数据(N, Me, ge, P)", type="xlsx")
        filenames['smoking'] = st.file_uploader("烟线(N, Me)", type="xlsx")
        filenames['operation'] = st.file_uploader("作业数据(N, Me)", type="xlsx")

data = loaddata(filenames)
model, SL, INT = calculate(data, MPTO, n_i_split, im, ηm, φ, rk, Gs, α, δ0, f, g, Hz)

tabs = st.tabs(["发动机外特性", "变矩器特性", "比油耗分布", '速度特性', "动力匹配", "交值数据", "整机特性", "车速分析", "高效区"])

with tabs[0]:
    st.markdown('### 发动机外特性曲线')
    plot('转速(r/min)', '扭矩(N*m)', plist=[
        [[SL['x_N'], SL['y_Me']], {'label':'仿真值'}],
        [[data['engine']['N'], data['engine']['Me'], 'o'], {'label':'实验值'}]
    ])

    st.markdown('### 附件扭矩消耗曲线')
    plot('转速(r/min)', '扭矩(N*m)', plist=[
        [[SL['x_N'], SL['y_Mba']], {'label':'仿真值'}],
        [[data['engine']['N'], data['engine']['Mba'], 'o'], {'label':'实验值'}]
    ])

with tabs[1]:
    st.markdown('### 变矩器特性曲线')
    plot('传动比', '扭矩(N*m)', plist=[
        [[INT['i'], INT['Mb1k']], {'label':'仿真值'}],
        [[data['converter']['i'], data['converter']['Mb'], 'o'], {'label':'实验值'}]
    ])

    st.markdown('### 变矩器K值曲线')
    plot('传动比', 'K值', plist=[
        [[INT['i'], INT['K']], {'label':'仿真值'}],
        [[data['converter']['i'], data['converter']['K'], 'o'], {'label':'实验值'}]
    ])

with tabs[2]:
    opdata = st.checkbox('作业数据')
    st.markdown('### 发动机比油耗分布耗图')
    levels = [190, 192, 195, 197, 200, 205, 210, 215, 225, 235, 255, 300, 350, 430]
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(8, 5)
    cs = plt.contour(
        data['consmap'].columns.values, 
        data['consmap'].index.values,
        data['consmap'].values, 
        levels=levels,
        extend='both',
        linewidths=(1,),
    )
    cs.changed()
    ax.set_xlabel("转速(r/min)")
    ax.set_ylabel("扭矩(N*m)")
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.clabel(cs, fmt='%.f', colors='b', fontsize=20)
    if opdata:
        ax.plot(data['operation']['N'], data['operation']['Me'], 'o')
    st.pyplot(fig)

    st.markdown('### 发动机比油耗模型误差')
    data['cons']['pred'] = model['ge'](data['cons'][['N', 'Me']].values)
    data['cons']['deviation'] = (data['cons']['pred'] - data['cons']['ge']).abs()
    data['cons']['rate'] = data['cons']['deviation'] / data['cons']['ge'] * 100
    st.write(f"\
        平均误差: {data['cons']['rate'].mean():.2f}%,\
        最大误差: {data['cons']['rate'].max():.2f}%\
    ")
    plot('比油耗(预测)(g/kw*h)', '比油耗(实验)(g/kw*h)', plist=[
        [[data['cons']['pred'], data['cons']['ge'], 'o'], dict(label='拟合曲线')],
    ])

with tabs[3]:
    st.markdown('### 发动机速度特性曲线')
    fig, ax = arrow_figure()

    ax.plot(data['engine']['N'], data['engine']['Me'], "o", label="实验值")
    p1, = ax.plot(SL['x_N'], SL['y_Me'], color='b', label="仿真值")
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.set_xlabel("转速(r/min)")
    ax.set_ylabel("扭矩(N*m)")
    ax.yaxis.label.set_color(p1.get_color())
    ax.legend()
    
    ax2 = add_secondary_axes(fig)
    p2, = ax2.plot(SL['x_N'], SL['y_ge'], color="r", label='比油耗(g/kw*h)')
    ax2.set_ylabel('比油耗(g/kw*h)')
    ax2.yaxis.label.set_color(p2.get_color())

    ax3 = add_secondary_axes(fig)
    p3, = ax3.plot(SL['x_N'], SL['y_Pb'], color="g", label='功率(kw/h)')
    ax3.set_ylabel('功率(kw/h)')
    ax3.yaxis.label.set_color(p3.get_color())

    st.pyplot(fig)


with tabs[4]:
    st.markdown('### 动力匹配图')
    plot('转速(r/min)', '扭矩(N*m)', plist=\
        [[[SL['x_N'], y_Mb], {}] for _, y_Mb in SL.filter(regex=r'y_Mb\[\d+\]').items()] + [
            [[SL['x_N'], SL['y_Mb']], {'label': 'Mb(发动机)'}],
            [[INT['N'], INT['Me'], 'o'], {'label': '交点', 'color': 'orange'}]
        ]
    )

with tabs[5]:
    st.markdown('### 匹配交值表')
    INTV0 = INT[meas]
    INTV0= INTV0.rename(columns={
        code: f"{ name } / {code} ({ unit })" for code, (name, unit) in meas.items()
    })
    st.download_button(label="下载交值表", 
        data=dataframe_to_bytes(INTV0),file_name='交值表.xlsx', mime='xls/xlsx')
    st.dataframe(INTV0.T.style.format("{:.2f}"), height=900)

with tabs[6]:
# 牵引效率, 牵引功率
    st.markdown('### 交值图表') 
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(label='x-轴', options=INTV0.columns, index=0)
    with col2:
        Y_axis = st.multiselect(label='y-轴', options=INTV0.columns, default=['加速度 / a (m/s^2)'])

    if Y_axis:
        INTV0 = INTV0.sort_values(x_axis)
        fig, ax = arrow_figure()
        ax.set_xlabel(x_axis)
        ax.grid()
        axs, ax_colors, ax_linestyle = {}, {}, {}
        for y_axis in Y_axis:
            unit = y_axis.split(' ')[-1]
            if unit not in axs:
                axs[unit] = add_secondary_axes(fig) if axs else ax
                axs[unit].set_ylabel(unit)
                ax_colors[unit] = colors[(len(axs) - 1) % len(colors)]
                axs[unit].yaxis.label.set_color(ax_colors[unit])
                ax_linestyle[unit] = 0
            else:
                ax_linestyle[unit] += 1
            plt.plot(INTV0[x_axis], INTV0[y_axis], label=y_axis, 
                color=ax_colors[unit], linestyle=linestyles[ax_linestyle[unit] % len(linestyles)])
        legend_shift = (len(axs) - 1) * 0.05
        # fig.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5 + legend_shift, -0.05), fancybox=True, shadow=True, ncol=3)
        st.pyplot(fig)


with tabs[7]:
    st.markdown('### 车速分析') 
    sr = st.slider("速度范围(km/h)", min_value=50, max_value=200, value=60, step=1) 
    Ff = INT['Ff'][0]
    n_points = int(len(INT) * 5. / 100.)
    reg = linregress(INT['Vt'][-n_points:], INT['Fk'][-n_points:])
    x_int = -reg.intercept / reg.slope
    vm = (Ff-reg.intercept) / reg.slope

    fig, ax = plot('理论行驶速度 Vt (km/h)', '切线牵引力 / Fk (KN)', plist=[
        [[INT['Vt'], INT['Fk']], {'label':'切线牵引力'}],
        [[range(sr), [Ff] * sr, '--'], {'label':'滚动阻力'}],
        [[[INT['Vt'].iloc[-1], x_int], [INT['Fk'].iloc[-1], 0], '--'], {'label':'延长线'}],
        [[vm,  Ff, 'o'], {'label':'交点'}],
    ], show=False)
    ax.annotate(f'最大车速: {vm:.1f}km/h', xy=(vm+1, Ff+1), xytext=(vm+5, Ff+10),
                arrowprops=dict(facecolor='black', shrink=0.01))
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.legend()
    st.pyplot(fig)

with tabs[8]:
    st.markdown('### 变矩器高效区') 
    eff_xi = intersect(INT['ηe'], INT['η'])
    fig, ax = plot('传动比', '传动效率(%)', plist=[
        [[INT['i'], INT['η']], {'label':'传动效率(%)'}],
        [[INT['i'], INT['ηe'], 'r--'], {'label':'高效区阈值(%)'}],
        [[INT['i'].iloc[eff_xi], INT['ηe'].iloc[eff_xi], 'o'], {'label':'交点', 'color': 'orange'}],
    ], show=False)
    ax.fill_between(
        INT['i'].iloc[eff_xi[0]: eff_xi[1]], 
        INT['η'].iloc[eff_xi[0]: eff_xi[1]], 
        INT['ηe'].iloc[eff_xi[0]: eff_xi[1]],
        color='orange', label='高效区',
    )
    ax.fill_between(
        INT['i'].iloc[eff_xi[2]:], 
        INT['η'].iloc[eff_xi[2]:], 
        INT['ηe'].iloc[eff_xi[2]:],
        color='orange',
    )
    ax.grid(linestyle = '--', linewidth = 0.5)
    ax.legend()
    st.pyplot(fig)