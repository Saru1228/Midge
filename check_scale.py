import swarm
import numpy as np
import os
from scipy.spatial import distance_matrix

folder = "/mnt/d/3Ddataset/"
if os.path.exists("Ob1.txt"):
    df = swarm._read_one_file("Ob1.txt")
    df = swarm.preprocess_full(df)
    
    # Get a frame with ~50-100 particles
    counts = df.groupby('t').size()
    target_t = counts[(counts > 40) & (counts < 80)].index[0]
    
    points = df[df['t'] == target_t][['x', 'y', 'z']].values
    
    print(f"Frame t={target_t}, Particles: {len(points)}")
    print(f"Coordinate Range: X[{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
    print(f"                  Y[{points[:,1].min():.1f}, {points[:,1].max():.1f}]")
    print(f"                  Z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
    
    # Calc Pairwise Distances
    dists = distance_matrix(points, points)
    # Exclude 0 (self-distance)
    dists[dists == 0] = np.nan
    
    min_dists = np.nanmin(dists, axis=1)
    print(f"Average Nearest Neighbor Dist: {np.nanmean(min_dists):.2f}")
    print(f"Max Nearest Neighbor Dist: {np.nanmax(min_dists):.2f}")
    
    # Check Delaunay Edge Lengths
    from scipy.spatial import Voronoi
    vor = Voronoi(points)
    edge_lengths = []
    for p1_idx, p2_idx in vor.ridge_points:
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        d = np.linalg.norm(p1 - p2)
        edge_lengths.append(d)
        
    print(f"Delaunay Edges: {len(edge_lengths)}")
    print(f"  Min Edge: {min(edge_lengths):.2f}")
    print(f"  Max Edge: {max(edge_lengths):.2f}")
    print(f"  Mean Edge: {np.mean(edge_lengths):.2f}")
    print(f"  Edges > 80mm: {sum(1 for d in edge_lengths if d > 80)}")
else:
    print("Ob1.txt not found for check.")
