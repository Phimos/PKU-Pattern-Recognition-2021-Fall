python cluster.py --data cluster_data/Aggregation.txt --algorithm kmeans
python cluster.py --data cluster_data/Aggregation.txt --algorithm kernel_kmeans --sigma 2
python cluster.py --data cluster_data/Aggregation.txt --algorithm ratio_cut --sigma 0.8
python cluster.py --data cluster_data/Aggregation.txt --algorithm ncut --sigma 0.8

python cluster.py --data cluster_data/jain.txt --algorithm kmeans
python cluster.py --data cluster_data/jain.txt --algorithm kernel_kmeans --sigma 2
python cluster.py --data cluster_data/jain.txt --algorithm ratio_cut --sigma 0.3
python cluster.py --data cluster_data/jain.txt --algorithm ncut --sigma 0.3

python cluster.py --data cluster_data/spiral.txt --algorithm kmeans
python cluster.py --data cluster_data/spiral.txt --algorithm kernel_kmeans --sigma 2
python cluster.py --data cluster_data/spiral.txt --algorithm ratio_cut --sigma 0.3
python cluster.py --data cluster_data/spiral.txt --algorithm ncut --sigma 0.3