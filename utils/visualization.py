import numpy as np
import plotly.graph_objects as go
import open3d as o3d

def read_obj(file):
    verts = []
    faces = []
    for line in file:
        if line.startswith('v '):  # Vertex
            verts.append([float(coord) for coord in line.strip().split()[1:]])
        elif line.startswith('f '):  # Face
            face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
            faces.append(face)
    return np.array(verts), np.array(faces)

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    fig = go.Figure(data=data,
                     layout=go.Layout(
                         updatemenus=[dict(type='buttons',
                                           showactive=False,
                                           y=1,
                                           x=0.8,
                                           xanchor='left',
                                           yanchor='bottom',
                                           pad=dict(t=45, r=10),
                                           buttons=[dict(label='Play',
                                                         method='animate',
                                                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                          transition=dict(duration=0),
                                                                          fromcurrent=True,
                                                                          mode='immediate'
                                                                          )]
                                                         )
                                                  ])]
                     ),
                     frames=frames
                     )
    return fig

def visualize_mesh(verts, faces):
    i, j, k = np.array(faces).T
    x, y, z = np.array(verts).T

    data = [go.Mesh3d(x=x, y=y, z=z, color='yellowgreen', opacity=0.50, i=i, j=j, k=k)]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                   line=dict(width=2,
                                             color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()