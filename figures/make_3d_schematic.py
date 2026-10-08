"""3D schematic illustration for this project.
Rendered in Python (matplotlib) - not ANSYS/Fluent/STAR-CCM+ output.
Run: python make_3d_schematic.py  (needs matplotlib, numpy)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def hull_mesh(L=100.0, B=20.0, T=8.0, D=6.0, p=1.0, q=0.8, nx=61, ny=31):
    xs = np.linspace(-L/2, L/2, nx)
    a = np.linspace(0, np.pi, ny)
    X, A = np.meshgrid(xs, a)
    xn = 2*X/L
    hb = (B/2.0)*np.maximum(0.0, 1-xn**2)**p
    Y = hb*np.cos(A)
    Z = D - (D+T)*np.sin(A)**q
    return X, Y, Z

def draw_hull(ax, L, B, T, D, p=1.0, q=0.8, color='#b8c4cc', alpha=0.95):
    X, Y, Z = hull_mesh(L, B, T, D, p, q)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=True,
                    rstride=2, cstride=2, linewidth=0, antialiased=True)

def box(ax, x0,x1, y0,y1, z0,z1, color='lightgray', alpha=1.0):
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],
             [v[0],v[1],v[5],v[4]],[v[2],v[3],v[7],v[6]],
             [v[1],v[2],v[6],v[5]],[v[4],v[7],v[3],v[0]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, alpha=alpha,
                                        edgecolor='#333333', linewidths=0.4))

def plane(ax, x0,x1, y0,y1, z, color, alpha):
    X, Y = np.meshgrid([x0,x1],[y0,y1])
    ax.plot_surface(X, Y, np.full_like(X,float(z)), color=color, alpha=alpha, shade=False)

def water(ax, x0,x1, y0,y1, z=0):
    plane(ax, x0,x1, y0,y1, z, '#7fb3d5', 0.16)

def seabed(ax, x0,x1, y0,y1, z, color='#d9c9a8'):
    plane(ax, x0,x1, y0,y1, z, color, 0.5)
    for gx in np.linspace(x0,x1,9):
        ax.plot([gx,gx],[y0,y1],[z,z], color='#b09a72', lw=0.4, alpha=0.6)
    for gy in np.linspace(y0,y1,7):
        ax.plot([x0,x1],[gy,gy],[z,z], color='#b09a72', lw=0.4, alpha=0.6)

def mooring_line(ax, p0, p1, n=60, **kw):
    s = np.linspace(0,1,n)
    x = p0[0]+(p1[0]-p0[0])*s
    y = p0[1]+(p1[1]-p0[1])*s
    z = p0[2]+(p1[2]-p0[2])*(s**1.6)
    ax.plot(x, y, z, **kw)

def finish(ax, fname, title, elev=18, azim=-58):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print('saved', fname)

from pathlib import Path
import numpy as np

L, B, T, D = 200.0, 30.0, 8.0, 10.0
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_subplot(111, projection='3d')

draw_hull(ax, L, B, T, D, p=0.8, q=0.9)
box(ax, -L/2, L/2, -B/2*0.92, B/2*0.92, D-0.5, D+0.5, color='#8a949c', alpha=0.9)
# ro-ro superstructure: long vehicle decks + bridge tower forward
box(ax, -80, 60, -12, 12, D, D+14, color='#dfe5ea', alpha=0.95)
box(ax, 62, 86, -10, 10, D, D+30, color='#cfd8dc', alpha=0.95)
water(ax, -220, 220, -160, 160)

# heave: vertical double arrow amidships
ax.quiver(0, 0, 25, 0, 0, 30, color='#c0392b', lw=2, arrow_length_ratio=0.18)
ax.quiver(0, 0, 25, 0, 0, -30, color='#c0392b', lw=2, arrow_length_ratio=0.18)
ax.text(6, 6, 58, 'heave', fontsize=10, color='#c0392b')
# pitch: arc arrow in x-z plane near bow
th = np.linspace(-0.5, 0.5, 40)
ax.plot(70+28*np.sin(th), np.zeros_like(th)+34, 30+28*(1-np.cos(th)), color='#1a6e2e', lw=2)
ax.quiver(70+28*np.sin(0.5), 34, 30+28*(1-np.cos(0.5)), 8, 0, 6, color='#1a6e2e', lw=2, arrow_length_ratio=0.3)
ax.text(104, 34, 46, 'pitch', fontsize=10, color='#1a6e2e')
# roll: arc arrow in y-z plane amidships
ph = np.linspace(-0.6, 0.6, 40)
ax.plot(np.zeros_like(ph)-30, 24*np.sin(ph), 34+24*(1-np.cos(ph)), color='#7d3c98', lw=2)
ax.quiver(-30, 24*np.sin(0.6), 34+24*(1-np.cos(0.6)), 0, 8, 6, color='#7d3c98', lw=2, arrow_length_ratio=0.3)
ax.text(-30, 34, 52, 'roll', fontsize=10, color='#7d3c98')
ax.text(-95, -40, D+40, 'ro-ro vessel', fontsize=10, ha='center')
ax.text2D(0.02, 0.03, 'Innovation: frozen inference from AIS observables\nP95 position error -54.6 m vs baseline (111 segments, 95% CI)\nNo motion sensors needed',
    transform=ax.transAxes, fontsize=8.5, va='bottom', ha='left',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88))
finish(ax, str(Path(__file__).parent / 'fig3_vessel_motion_3d.png'),
       'Ro-ro vessel motion schematic - heave / pitch / roll', elev=14, azim=-64)
