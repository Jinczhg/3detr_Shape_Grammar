import numpy as np
from plotly import graph_objects as go

pc = []
with open('/home/jzhang72/NetBeansProjects/sharp-kt/dataset_room/0/pointcloud.txt', 'r') as f:
    for line in f.readlines():
        point = line.split(' ')
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        point_as_array = [x, y, z]
        pc.append(point_as_array)
pc = np.asarray(pc)
# pc[pc == -30] = -10

fig = go.Figure(data=[
    go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                 mode='markers',
                 marker=dict(
                     size=0.5,
                     color=-100 * z,  # set color to an array/list of desired values
                     colorscale='Blues',  # choose a colorscale
                     opacity=0.8
                 )
                 ), ])
fig.show()
