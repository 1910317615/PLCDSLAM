import os

def MakeDir(nd):
  if not os.path.exists(nd):
    os.mkdir(nd)


dataset_idx = 0
datasets = ["tartanair"]
dataset = datasets[dataset_idx]

workspace = "/home/lan/Airslam_ws"


if "tartanair" in dataset:
  dataroot = "/home/lan/桌面/tartanair/ocean"
  saving_root = "/home/lan/桌面/临时"
  launch_file = "vo_tartanair.launch"

else:
  print("{} is not support".format(dataset_idx))


print(dataset_idx)
print(dataroot)

MakeDir(saving_root)
map_root = os.path.join(saving_root, "maps")
MakeDir(map_root)
sequences = os.listdir(dataroot)
# sequences = ["MH_03_medium"]
for sequence in sequences:
  seq_dataroot = os.path.join(dataroot, sequence)
  seq_save_root = os.path.join(map_root, sequence)
  MakeDir(seq_save_root)
  os.system("cd {} & roslaunch air_slam {} dataroot:={} saving_dir:={} visualization:=false".format(workspace, launch_file, seq_dataroot, seq_save_root))
