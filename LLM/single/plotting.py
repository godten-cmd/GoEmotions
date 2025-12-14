# import plotly.graph_objs as go
# import plotly.io as pio

# # VS Code에서 자동 브라우저 실행 안 될 경우 'notebook' 대신 'browser' 사용 가능
# pio.renderers.default = 'iframe_connected'  # 또는 'vscode', 'notebook'

# # 예제 데이터
# x = [0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 1, 1, 1, 1, 1]
# y = [0, 0.25, 0.5, 0.75, 1, 0, 0.25, 0.5, 0.75, 1, 0, 0.25, 0.5, 0.75, 1, 0, 0.25, 0.5, 0.75, 1, 0, 0.25, 0.5, 0.75, 1]
# z = [0.4436318083455104, 0.44597022569871464, 0.44303576930811767, 0.44557519966778153, 0.441886528129043, 0.443239947278968, 0.4442975072495047, 0.44417058745775756, 0.4463184688762758, 0.4447196284409029, 0.4395063363364545, 0.44383884441478977, 0.4446410075304853, 0.44397521317720884, 0.4459398944636561, 0.4449752622544293, 0.43979438282732913, 0.437527150978279, 0.44445195933725096, 0.4359520547324275, 0.44484983639136566, 0.44511127583342525, 0.4412538152664426, 0.4398362208222606, 0.4294634143290542]

# # 3D 산점도 생성
# fig = go.Figure(data=[go.Scatter3d(
#     x=x,
#     y=y,
#     z=z,
#     mode='markers',  # 점 + 선
#     marker=dict(
#         size=8,
#         color=z,              # 색상에 깊이감 주기
#         colorscale='Viridis', # 색상 스케일
#         opacity=0.8
#     )
# )])

# # 레이아웃 설정 (옵션)
# fig.update_layout(
#     scene=dict(
#         xaxis_title='X 축',
#         yaxis_title='Y 축',
#         zaxis_title='Z 축'
#     ),
#     title='인터랙티브 3D 플롯'
# )
# fig.write_html("plot.html", auto_open=True)

# import plotly.graph_objects as go
# import numpy as np

# # 원본 데이터 (1차원 리스트)
# x = [0, 0, 0, 0, 0, 
#      0.25, 0.25, 0.25, 0.25, 0.25, 
#      0.5, 0.5, 0.5, 0.5, 0.5, 
#      0.75, 0.75, 0.75, 0.75, 0.75, 
#      1, 1, 1, 1, 1]

# y = [0, 0.25, 0.5, 0.75, 1,
#      0, 0.25, 0.5, 0.75, 1,
#      0, 0.25, 0.5, 0.75, 1,
#      0, 0.25, 0.5, 0.75, 1,
#      0, 0.25, 0.5, 0.75, 1]

# z = [0.44363, 0.44597, 0.44303, 0.44558, 0.44189, 
#      0.44324, 0.44430, 0.44417, 0.44632, 0.44472, 
#      0.43951, 0.44384, 0.44464, 0.44398, 0.44594, 
#      0.44498, 0.43979, 0.43753, 0.44445, 0.43595, 
#      0.44485, 0.44511, 0.44125, 0.43984, 0.42946]

# # 2D 격자로 변환 (5 x 5)
# Z = np.array(z).reshape(5, 5).T
# X = np.array(x).reshape(5, 5).T
# Y = np.array(y).reshape(5, 5).T

# # Plotly 등고선 그래프
# fig = go.Figure(data=go.Contour(
#     z=Z,
#     x=X[0],   # x 좌표값 (열 방향)
#     y=Y[:,0], # y 좌표값 (행 방향)
#     colorscale='Viridis',
#     contours=dict(
#         coloring='heatmap',
#         showlabels=True,  # 등고선 값 표시
#         labelfont=dict(size=12, color='white')
#     ),
#     colorbar=dict(title='Z 값'),
# ))

# fig.update_layout(
#     title='인터랙티브 등고선 그래프',
#     xaxis_title='X',
#     yaxis_title='Y'
# )

# # 브라우저에서 보기
# fig.write_html("contour_plot.html", auto_open=True)

import plotly.graph_objects as go
import numpy as np

# 원본 데이터 (1차원 리스트)
x = [0, 0, 0, 0, 0, 
     0.25, 0.25, 0.25, 0.25, 0.25, 
     0.5, 0.5, 0.5, 0.5, 0.5, 
     0.75, 0.75, 0.75, 0.75, 0.75, 
     1, 1, 1, 1, 1]

y = [0, 0.25, 0.5, 0.75, 1,
     0, 0.25, 0.5, 0.75, 1,
     0, 0.25, 0.5, 0.75, 1,
     0, 0.25, 0.5, 0.75, 1,
     0, 0.25, 0.5, 0.75, 1]

z = [0.4548, 0.4548, 0.4536, 0.4572, 0.4528, 0.4548, 0.4572, 0.456, 0.458, 0.4548, 0.4532, 0.4552, 0.4516, 0.4544, 0.456, 0.4544, 0.4532, 0.4536, 0.4528, 0.45, 0.456, 0.458, 0.4528, 0.4488, 0.4496]
# 5x5 격자로 변환
Z = np.array(z).reshape(5, 5).T
X = np.array(x).reshape(5, 5).T
Y = np.array(y).reshape(5, 5).T

# 3D 등고선 Surface
surface = go.Surface(
    z=Z,
    x=X,
    y=Y,
    colorscale='Viridis',
    opacity=0.9,
    contours={
        "z": {
            "show": True,
            "start": np.min(Z),
            "end": np.max(Z),
            "size": 0.0015,
            "color": "white",
            "width": 2
        }
    },
    showscale=True
)

# 3D 산점도
scatter = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        opacity=0.9
    ),
    name='Data Points'
)

# 두 trace를 한 Figure에 넣기
fig = go.Figure(data=[surface, scatter])

fig.update_layout(
    title="3D 등고선 + 산점도",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    )
)

# 브라우저에서 보기
fig.write_html("3d_contour_with_scatter-1.html", auto_open=True)
